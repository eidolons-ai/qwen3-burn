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
                [0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..head_dim],
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

        self.cache
            .clone()
            .slice([0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..head_dim])
    }

    /// Returns the current cached sequence length.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.cur_seq_len
    }
}
