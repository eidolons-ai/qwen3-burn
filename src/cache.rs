use burn::tensor::{backend::Backend, Device, Tensor};

/// Key-value cache for autoregressive generation.
///
/// Stores cached key or value tensors with shape `[batch_size, num_heads, seq_len, head_dim]`.
pub struct KvCache<B: Backend> {
    cache: Tensor<B, 4>,
    max_seq_len: usize,
    cur_seq_len: usize,
}

impl<B: Backend> KvCache<B> {
    /// Creates a new empty cache.
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            cache: Tensor::empty([batch_size, num_heads, max_seq_len, head_dim], device),
            max_seq_len,
            cur_seq_len: 0,
        }
    }

    /// Reset the cache state.
    pub fn reset(&mut self) {
        self.cache = Tensor::empty(self.cache.shape(), &self.cache.device());
        self.cur_seq_len = 0;
    }

    /// Append new key/value tensor to the cache and return the accumulated result.
    pub fn forward(&mut self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, num_heads, seq_len, head_dim] = tensor.dims();

        // If the incoming chunk fills or exceeds the cache, just keep the last max_seq_len tokens
        if seq_len >= self.max_seq_len {
            let start = seq_len - self.max_seq_len;
            let truncated =
                tensor.slice([0..batch_size, 0..num_heads, start..seq_len, 0..head_dim]);
            self.cache = self.cache.clone().slice_assign(
                [
                    0..batch_size,
                    0..num_heads,
                    0..self.max_seq_len,
                    0..head_dim,
                ],
                truncated,
            );
            self.cur_seq_len = self.max_seq_len;
            return self.cache.clone().slice([
                0..batch_size,
                0..num_heads,
                0..self.max_seq_len,
                0..head_dim,
            ]);
        }

        let mut new_seq_len = self.cur_seq_len + seq_len;

        // If we exceed max_seq_len, shift the cache (sliding window)
        if new_seq_len > self.max_seq_len {
            self.cur_seq_len = self.max_seq_len - seq_len;
            let prev_slice = self.cache.clone().slice([
                0..batch_size,
                0..num_heads,
                seq_len..self.max_seq_len,
                0..head_dim,
            ]);
            self.cache = self.cache.clone().slice_assign(
                [
                    0..batch_size,
                    0..num_heads,
                    0..self.cur_seq_len,
                    0..head_dim,
                ],
                prev_slice,
            );
            new_seq_len = self.max_seq_len;
        }

        self.cache = self.cache.clone().slice_assign(
            [
                0..batch_size,
                0..num_heads,
                self.cur_seq_len..new_seq_len,
                0..head_dim,
            ],
            tensor,
        );

        self.cur_seq_len += seq_len;

        self.cache.clone().slice([
            0..batch_size,
            0..num_heads,
            0..self.cur_seq_len,
            0..head_dim,
        ])
    }

    /// Returns the current cached sequence length.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.cur_seq_len
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

    #[test]
    fn cache_initial_state() {
        let cache = KvCache::<B>::new(1, 2, 8, 4, &device());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_single_append() {
        let dev = device();
        let mut cache = KvCache::<B>::new(1, 2, 8, 4, &dev);
        let tensor = Tensor::<B, 4>::ones([1, 2, 3, 4], &dev);
        let out = cache.forward(tensor);
        assert_eq!(cache.len(), 3);
        assert_eq!(out.dims(), [1, 2, 3, 4]);
    }

    #[test]
    fn cache_multiple_appends() {
        let dev = device();
        let mut cache = KvCache::<B>::new(1, 2, 16, 4, &dev);

        // Append 3 tokens
        let t1 = Tensor::<B, 4>::ones([1, 2, 3, 4], &dev);
        let out1 = cache.forward(t1);
        assert_eq!(cache.len(), 3);
        assert_eq!(out1.dims(), [1, 2, 3, 4]);

        // Append 2 more tokens
        let t2 = Tensor::<B, 4>::ones([1, 2, 2, 4], &dev) * 2.0;
        let out2 = cache.forward(t2);
        assert_eq!(cache.len(), 5);
        assert_eq!(out2.dims(), [1, 2, 5, 4]);
    }

    #[test]
    fn cache_preserves_values() {
        let dev = device();
        let mut cache = KvCache::<B>::new(1, 1, 8, 2, &dev);

        // Append [1.0, 2.0]
        let t1 = Tensor::<B, 4>::from_floats([[[[1.0, 2.0]]]], &dev);
        cache.forward(t1);

        // Append [3.0, 4.0]
        let t2 = Tensor::<B, 4>::from_floats([[[[3.0, 4.0]]]], &dev);
        let out = cache.forward(t2);

        assert_eq!(cache.len(), 2);
        let vals: Vec<f32> = out.to_data().iter::<f32>().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cache_reset() {
        let dev = device();
        let mut cache = KvCache::<B>::new(1, 2, 8, 4, &dev);
        let tensor = Tensor::<B, 4>::ones([1, 2, 3, 4], &dev);
        cache.forward(tensor);
        assert_eq!(cache.len(), 3);

        cache.reset();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_sliding_window() {
        let dev = device();
        let mut cache = KvCache::<B>::new(1, 1, 4, 2, &dev); // max_seq_len=4

        // Fill to capacity: 4 tokens
        let t =
            Tensor::<B, 4>::from_floats([[[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]]], &dev);
        let out = cache.forward(t);
        assert_eq!(cache.len(), 4);
        assert_eq!(out.dims(), [1, 1, 4, 2]);

        // Append 1 more: should trigger sliding window
        let t = Tensor::<B, 4>::from_floats([[[[5.0, 0.0]]]], &dev);
        let out = cache.forward(t);
        assert_eq!(cache.len(), 4); // still max_seq_len
        assert_eq!(out.dims(), [1, 1, 4, 2]);

        // Values should be [2,3,4,5] (oldest dropped)
        let vals: Vec<f32> = out.to_data().iter::<f32>().collect();
        assert_eq!(vals[0], 2.0); // first token's first element
        assert_eq!(vals[2], 3.0);
        assert_eq!(vals[4], 4.0);
        assert_eq!(vals[6], 5.0);
    }

    #[test]
    fn cache_chunk_larger_than_max_seq_len() {
        let dev = device();
        let mut cache = KvCache::<B>::new(1, 1, 3, 2, &dev); // max_seq_len=3

        // Append 5 tokens into a cache of size 3 — should keep last 3
        let t = Tensor::<B, 4>::from_floats(
            [[[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]]],
            &dev,
        );
        let out = cache.forward(t);
        assert_eq!(cache.len(), 3);
        assert_eq!(out.dims(), [1, 1, 3, 2]);
        let vals: Vec<f32> = out.to_data().iter::<f32>().collect();
        assert_eq!(vals[0], 3.0);
        assert_eq!(vals[2], 4.0);
        assert_eq!(vals[4], 5.0);
    }

    #[test]
    fn cache_chunk_exactly_max_seq_len() {
        let dev = device();
        let mut cache = KvCache::<B>::new(1, 1, 3, 2, &dev); // max_seq_len=3

        // Append exactly 3 tokens into cache of size 3
        let t = Tensor::<B, 4>::from_floats([[[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]], &dev);
        let out = cache.forward(t);
        assert_eq!(cache.len(), 3);
        assert_eq!(out.dims(), [1, 1, 3, 2]);
        let vals: Vec<f32> = out.to_data().iter::<f32>().collect();
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[2], 2.0);
        assert_eq!(vals[4], 3.0);
    }
}
