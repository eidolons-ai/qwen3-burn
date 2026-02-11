use burn::tensor::{backend::Backend, Int, Tensor};
use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};

/// Token sampling strategy.
pub enum Sampler {
    TopP(Box<TopP>),
    Argmax,
}

impl Sampler {
    /// Create a new top-p (nucleus) sampler.
    pub fn new_top_p(p: f64, seed: u64) -> Self {
        Self::TopP(Box::new(TopP::new(p, seed)))
    }

    /// Sample the next token from logits with shape `[1, vocab_size]`.
    pub fn sample<B: Backend>(&mut self, logits: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        match self {
            Self::TopP(s) => s.sample(logits),
            Self::Argmax => logits.argmax(1),
        }
    }

    /// Sample from a pre-computed f64 probability distribution on CPU.
    ///
    /// This avoids running softmax on the GPU where f16 precision can cause
    /// overflow/underflow over large vocabularies.
    pub fn sample_probs(&mut self, probs: &[f64]) -> u32 {
        match self {
            Self::TopP(s) => s.sample_probs(probs),
            Self::Argmax => probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0 as u32,
        }
    }
}

/// Top-p (nucleus) sampling selects from the smallest set of tokens whose
/// cumulative probability exceeds the threshold p.
pub struct TopP {
    p: f64,
    rng: StdRng,
}

impl TopP {
    pub fn new(p: f64, seed: u64) -> Self {
        Self {
            p,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn sample<B: Backend>(&mut self, probs: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        assert_eq!(
            probs.dims()[0],
            1,
            "Top-p sampling only supports batch size 1"
        );
        let (probs_sort, probs_idx) = probs.sort_descending_with_indices(1);

        let mut probs_sort = probs_sort.to_data().iter::<f64>().collect::<Vec<_>>();

        let mut cumsum = 0.0;
        probs_sort.iter_mut().for_each(|x| {
            if cumsum >= self.p {
                *x = 0.0;
            } else {
                cumsum += *x;
            }
        });

        let next_token_idx = WeightedIndex::new(probs_sort)
            .unwrap()
            .sample(&mut self.rng);

        probs_idx.slice([0..1, next_token_idx..next_token_idx + 1])
    }

    /// Sample from f64 probabilities on CPU — avoids f16 softmax precision issues.
    pub fn sample_probs(&mut self, probs: &[f64]) -> u32 {
        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let mut weights: Vec<f64> = Vec::with_capacity(indexed.len());
        let mut cumsum = 0.0;
        for &(_, p) in &indexed {
            if cumsum >= self.p {
                weights.push(0.0);
            } else {
                cumsum += p;
                weights.push(p);
            }
        }

        let next_token_idx = WeightedIndex::new(weights)
            .unwrap()
            .sample(&mut self.rng);

        indexed[next_token_idx].0 as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn device() -> <B as Backend>::Device {
        Default::default()
    }

    fn sample_id(sampler: &mut Sampler, probs: Tensor<B, 2>) -> i64 {
        let out = sampler.sample(probs);
        out.to_data().iter::<i64>().next().unwrap()
    }

    #[test]
    fn argmax_picks_highest() {
        let dev = device();
        let mut sampler = Sampler::Argmax;
        // Token 2 has highest probability
        let probs = Tensor::<B, 2>::from_floats([[0.1, 0.2, 0.5, 0.15, 0.05]], &dev);
        assert_eq!(sample_id(&mut sampler, probs), 2);
    }

    #[test]
    fn argmax_first_on_tie() {
        let dev = device();
        let mut sampler = Sampler::Argmax;
        let probs = Tensor::<B, 2>::from_floats([[0.5, 0.5, 0.0]], &dev);
        // argmax returns first occurrence
        assert_eq!(sample_id(&mut sampler, probs), 0);
    }

    #[test]
    fn argmax_deterministic() {
        let dev = device();
        let mut sampler = Sampler::Argmax;
        let probs = Tensor::<B, 2>::from_floats([[0.1, 0.3, 0.6]], &dev);
        // Same input should always give same output
        let r1 = sample_id(&mut sampler, probs.clone());
        let r2 = sample_id(&mut sampler, probs);
        assert_eq!(r1, r2);
        assert_eq!(r1, 2);
    }

    #[test]
    fn top_p_returns_valid_index() {
        let dev = device();
        let mut sampler = Sampler::new_top_p(0.9, 42);
        let probs = Tensor::<B, 2>::from_floats([[0.1, 0.2, 0.3, 0.4]], &dev);
        let idx = sample_id(&mut sampler, probs);
        assert!((0..4).contains(&idx));
    }

    #[test]
    fn top_p_with_p_1_can_sample_any() {
        // With p=1.0, all tokens are candidates
        let dev = device();
        let probs = Tensor::<B, 2>::from_floats([[0.25, 0.25, 0.25, 0.25]], &dev);
        // Run multiple samples, verify all are in range
        let mut sampler = Sampler::new_top_p(1.0, 123);
        for _ in 0..20 {
            let idx = sample_id(&mut sampler, probs.clone());
            assert!((0..4).contains(&idx), "index {} out of range", idx);
        }
    }

    #[test]
    fn top_p_concentrates_on_top() {
        // With very low p, should almost always pick the highest-probability token
        let dev = device();
        let probs = Tensor::<B, 2>::from_floats([[0.01, 0.01, 0.01, 0.97]], &dev);
        let mut sampler = Sampler::new_top_p(0.1, 42);
        for _ in 0..10 {
            let idx = sample_id(&mut sampler, probs.clone());
            assert_eq!(
                idx, 3,
                "with p=0.1 and 0.97 mass on token 3, should always pick 3"
            );
        }
    }
}
