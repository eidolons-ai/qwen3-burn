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
}
