use burn::module::Param;
use burn::nn::Embedding;
use burn::prelude::*;
use burn::tensor::activation;

use crate::cache::KvCache;

/// RMS Layer Normalization.
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(size: usize, eps: f64, device: &Device<B>) -> Self {
        let weight = Tensor::ones([size], device);
        Self {
            weight: Param::from_tensor(weight),
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let variance = x.clone().powf_scalar(2.0).mean_dim(2);
        let normed = x * (variance + self.eps).sqrt().recip();
        normed * self.weight.val().unsqueeze::<3>()
    }
}

/// Rotary Position Embedding (RoPE).
pub struct RotaryEmbedding<B: Backend> {
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Create RoPE tables for the given max sequence length.
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device<B>) -> Self {
        let half_dim = head_dim / 2;

        // Compute inverse frequencies: theta^(-2i/d) for i in 0..half_dim
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (theta.powf(i as f64 * 2.0 / head_dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device);

        // Position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::<B, 1>::from_floats(positions.as_slice(), device);

        // Outer product: [max_seq_len] x [half_dim] -> [max_seq_len, half_dim]
        let freqs = positions
            .unsqueeze::<2>()
            .transpose()
            .matmul(inv_freq.unsqueeze::<2>());

        // Duplicate to full head_dim: [max_seq_len, head_dim]
        let freqs = Tensor::cat(vec![freqs.clone(), freqs], 1);

        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        Self { cos, sin }
    }

    /// Apply rotary embeddings to query and key tensors.
    ///
    /// Input shapes: `[batch, num_heads, seq_len, head_dim]`
    /// `start_pos` is the position offset for cached generation.
    pub fn apply(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        start_pos: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_batch, _heads, seq_len, _dim] = q.dims();

        // Slice cos/sin for the current positions: [seq_len, head_dim]
        // Then unsqueeze to [1, 1, seq_len, head_dim] for broadcasting with [batch, heads, seq_len, head_dim]
        let cos = self
            .cos
            .clone()
            .slice([start_pos..start_pos + seq_len])
            .unsqueeze::<3>()
            .unsqueeze::<4>();
        let sin = self
            .sin
            .clone()
            .slice([start_pos..start_pos + seq_len])
            .unsqueeze::<3>()
            .unsqueeze::<4>();

        let q_embed = q.clone() * cos.clone() + rotate_half(q) * sin.clone();
        let k_embed = k.clone() * cos + rotate_half(k) * sin;

        (q_embed, k_embed)
    }
}

/// Rotate the second half of the last dimension: [-x2, x1]
fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, seq_len, dim] = x.dims();
    let half = dim / 2;
    let x1 = x.clone().slice([0..batch, 0..heads, 0..seq_len, 0..half]);
    let x2 = x.slice([0..batch, 0..heads, 0..seq_len, half..dim]);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

/// Multi-head attention with grouped-query attention and QK-Norm.
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    q_proj: nn::Linear<B>,
    k_proj: nn::Linear<B>,
    v_proj: nn::Linear<B>,
    o_proj: nn::Linear<B>,
    q_norm: RmsNorm<B>,
    k_norm: RmsNorm<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

/// Grouped multi-head attention key-value cache pair.
pub struct AttentionKvCache<B: Backend> {
    pub k_cache: KvCache<B>,
    pub v_cache: KvCache<B>,
}

impl<B: Backend> AttentionKvCache<B> {
    pub fn new(
        batch_size: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            k_cache: KvCache::new(batch_size, num_kv_heads, max_seq_len, head_dim, device),
            v_cache: KvCache::new(batch_size, num_kv_heads, max_seq_len, head_dim, device),
        }
    }

    pub fn reset(&mut self) {
        self.k_cache.reset();
        self.v_cache.reset();
    }
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        device: &Device<B>,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        Self {
            q_proj: nn::LinearConfig::new(d_model, q_dim)
                .with_bias(false)
                .init(device),
            k_proj: nn::LinearConfig::new(d_model, kv_dim)
                .with_bias(false)
                .init(device),
            v_proj: nn::LinearConfig::new(d_model, kv_dim)
                .with_bias(false)
                .init(device),
            o_proj: nn::LinearConfig::new(q_dim, d_model)
                .with_bias(false)
                .init(device),
            q_norm: RmsNorm::new(head_dim, rms_norm_eps, device),
            k_norm: RmsNorm::new(head_dim, rms_norm_eps, device),
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    /// Forward pass.
    ///
    /// - `x`: `[batch, seq_len, d_model]`
    /// - `rope`: Rotary position embeddings
    /// - `mask`: Causal attention mask `[seq_len, total_seq_len]`
    /// - `cache`: Optional KV cache for autoregressive generation
    /// - `start_pos`: Position offset for RoPE
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        mask: Option<Tensor<B, 2>>,
        cache: &mut AttentionKvCache<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _d_model] = x.dims();

        // Project Q, K, V
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, num_heads, head_dim]
        let q = q.reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);

        // QK-Norm: apply RMSNorm per-head on the head_dim dimension
        let q = self.q_norm.forward(q.reshape([batch * seq_len * self.num_heads, 1, self.head_dim]))
            .reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = self.k_norm.forward(k.reshape([batch * seq_len * self.num_kv_heads, 1, self.head_dim]))
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, heads, seq_len, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Apply RoPE
        let (q, k) = rope.apply(q, k, start_pos);

        // Update KV cache
        let k = cache.k_cache.forward(k);
        let v = cache.v_cache.forward(v);

        // GQA: repeat K,V heads to match Q heads
        let num_kv_groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, num_kv_groups);
        let v = repeat_kv(v, num_kv_groups);

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt().recip();
        let attn_weights = q.matmul(k.transpose()) * scale;

        // Apply causal mask: mask is [q_seq, kv_seq], expand to [1, 1, q_seq, kv_seq]
        let attn_weights = if let Some(mask) = mask {
            attn_weights + mask.unsqueeze::<3>().unsqueeze::<4>()
        } else {
            attn_weights
        };

        let attn_weights = activation::softmax(attn_weights, 3);

        // Weighted sum over values
        let attn_output = attn_weights.matmul(v);

        // Reshape back to [batch, seq_len, d_model]
        let attn_output = attn_output
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(attn_output)
    }
}

/// Repeat KV heads for grouped-query attention.
/// Input: `[batch, num_kv_heads, seq_len, head_dim]`
/// Output: `[batch, num_kv_heads * n_rep, seq_len, head_dim]`
fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
    if n_rep == 1 {
        return x;
    }
    let [batch, num_kv_heads, seq_len, head_dim] = x.dims();
    x.unsqueeze_dim::<5>(2)
        .expand([batch, num_kv_heads, n_rep, seq_len, head_dim])
        .reshape([batch, num_kv_heads * n_rep, seq_len, head_dim])
}

/// SwiGLU feed-forward network.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    gate_proj: nn::Linear<B>,
    up_proj: nn::Linear<B>,
    down_proj: nn::Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(d_model: usize, intermediate_size: usize, device: &Device<B>) -> Self {
        Self {
            gate_proj: nn::LinearConfig::new(d_model, intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: nn::LinearConfig::new(d_model, intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: nn::LinearConfig::new(intermediate_size, d_model)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// Single transformer decoder block.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    input_layernorm: RmsNorm<B>,
    self_attn: MultiHeadAttention<B>,
    post_attention_layernorm: RmsNorm<B>,
    mlp: FeedForward<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_norm_eps: f64,
        device: &Device<B>,
    ) -> Self {
        Self {
            input_layernorm: RmsNorm::new(d_model, rms_norm_eps, device),
            self_attn: MultiHeadAttention::new(
                d_model,
                num_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                device,
            ),
            post_attention_layernorm: RmsNorm::new(d_model, rms_norm_eps, device),
            mlp: FeedForward::new(d_model, intermediate_size, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        mask: Option<Tensor<B, 2>>,
        cache: &mut AttentionKvCache<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        // Pre-norm attention with residual
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let x = self.self_attn.forward(x, rope, mask, cache, start_pos);
        let x = x + residual;

        // Pre-norm FFN with residual
        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(x);
        x + residual
    }
}

/// Full Qwen3 transformer model.
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<TransformerBlock<B>>,
    norm: RmsNorm<B>,
    lm_head: nn::Linear<B>,
}

impl<B: Backend> Transformer<B> {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_norm_eps: f64,
        tie_word_embeddings: bool,
        device: &Device<B>,
    ) -> Self {
        let embed_tokens = nn::EmbeddingConfig::new(vocab_size, d_model).init(device);

        let layers = (0..num_layers)
            .map(|_| {
                TransformerBlock::new(
                    d_model,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    intermediate_size,
                    rms_norm_eps,
                    device,
                )
            })
            .collect();

        let norm = RmsNorm::new(d_model, rms_norm_eps, device);

        let lm_head = if tie_word_embeddings {
            // Will be overwritten with embed_tokens weight during loading
            nn::LinearConfig::new(d_model, vocab_size)
                .with_bias(false)
                .init(device)
        } else {
            nn::LinearConfig::new(d_model, vocab_size)
                .with_bias(false)
                .init(device)
        };

        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        }
    }

    /// Forward pass through the transformer.
    ///
    /// - `tokens`: `[batch, seq_len]` token IDs
    /// - `rope`: Rotary position embeddings
    /// - `mask`: Optional causal mask
    /// - `caches`: KV caches for each layer
    /// - `start_pos`: Position offset
    ///
    /// Returns logits `[batch, seq_len, vocab_size]`.
    pub fn forward(
        &self,
        tokens: Tensor<B, 2, Int>,
        rope: &RotaryEmbedding<B>,
        mask: Option<Tensor<B, 2>>,
        caches: &mut [AttentionKvCache<B>],
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let mut x = self.embed_tokens.forward(tokens);

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x, rope, mask.clone(), &mut caches[i], start_pos);
        }

        x = self.norm.forward(x);
        self.lm_head.forward(x)
    }

    // --- Weight loading methods ---

    pub fn load_embed_tokens(mut self, weight: Tensor<B, 2>) -> Self {
        self.embed_tokens.weight = Param::from_tensor(weight);
        self
    }

    pub fn load_norm(mut self, weight: Tensor<B, 1>) -> Self {
        self.norm.weight = Param::from_tensor(weight);
        self
    }

    pub fn load_lm_head(mut self, weight: Tensor<B, 2>) -> Self {
        self.lm_head.weight = Param::from_tensor(weight);
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_layer(
        mut self,
        layer_idx: usize,
        q_proj_w: Tensor<B, 2>,
        k_proj_w: Tensor<B, 2>,
        v_proj_w: Tensor<B, 2>,
        o_proj_w: Tensor<B, 2>,
        q_norm_w: Tensor<B, 1>,
        k_norm_w: Tensor<B, 1>,
        gate_proj_w: Tensor<B, 2>,
        up_proj_w: Tensor<B, 2>,
        down_proj_w: Tensor<B, 2>,
        input_ln_w: Tensor<B, 1>,
        post_attn_ln_w: Tensor<B, 1>,
    ) -> Self {
        let layer = &mut self.layers[layer_idx];

        // Attention projections
        layer.self_attn.q_proj.weight = Param::from_tensor(q_proj_w);
        layer.self_attn.k_proj.weight = Param::from_tensor(k_proj_w);
        layer.self_attn.v_proj.weight = Param::from_tensor(v_proj_w);
        layer.self_attn.o_proj.weight = Param::from_tensor(o_proj_w);

        // QK-Norm
        layer.self_attn.q_norm.weight = Param::from_tensor(q_norm_w);
        layer.self_attn.k_norm.weight = Param::from_tensor(k_norm_w);

        // MLP
        layer.mlp.gate_proj.weight = Param::from_tensor(gate_proj_w);
        layer.mlp.up_proj.weight = Param::from_tensor(up_proj_w);
        layer.mlp.down_proj.weight = Param::from_tensor(down_proj_w);

        // Layer norms
        layer.input_layernorm.weight = Param::from_tensor(input_ln_w);
        layer.post_attention_layernorm.weight = Param::from_tensor(post_attn_ln_w);

        self
    }
}

/// Build a causal attention mask of shape `[seq_len, total_seq_len]`.
/// Positions that should not be attended to get `f32::NEG_INFINITY`.
pub fn build_causal_mask<B: Backend>(
    seq_len: usize,
    total_seq_len: usize,
    device: &Device<B>,
) -> Tensor<B, 2> {
    // Create upper triangular mask: mask[i][j] = -inf if j > i + offset
    let offset = total_seq_len - seq_len;
    let mut mask_data = vec![0.0f32; seq_len * total_seq_len];
    for i in 0..seq_len {
        for j in 0..total_seq_len {
            if j > i + offset {
                mask_data[i * total_seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::<B, 1>::from_floats(&mask_data[..], device).reshape([seq_len, total_seq_len])
}
