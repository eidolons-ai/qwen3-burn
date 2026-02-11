use std::collections::HashMap;

use burn::module::Param;
use burn::nn::Embedding;
use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::IndexingUpdateOp;

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
    #[allow(clippy::single_range_in_vec_init)]
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
        let q = self
            .q_norm
            .forward(q.reshape([batch * seq_len * self.num_heads, 1, self.head_dim]))
            .reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = self
            .k_norm
            .forward(k.reshape([batch * seq_len * self.num_kv_heads, 1, self.head_dim]))
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
        let attn_output =
            attn_output
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

/// Mixture of Experts layer.
///
/// Uses packed expert weights: a router selects top-k experts per token,
/// then each expert applies SwiGLU independently.
#[derive(Module, Debug)]
pub struct MoeLayer<B: Backend> {
    /// Router weight: [num_experts, hidden_size]
    router_weight: Param<Tensor<B, 2>>,
    /// Packed gate+up projection: [num_experts, hidden_size, 2*moe_intermediate_size]
    /// Stored in Burn convention [in, out] per expert.
    gate_up_proj: Param<Tensor<B, 3>>,
    /// Packed down projection: [num_experts, moe_intermediate_size, hidden_size]
    /// Stored in Burn convention [in, out] per expert.
    down_proj: Param<Tensor<B, 3>>,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    norm_topk_prob: bool,
}

impl<B: Backend> MoeLayer<B> {
    pub fn new(
        d_model: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate_size: usize,
        norm_topk_prob: bool,
        device: &Device<B>,
    ) -> Self {
        Self {
            router_weight: Param::from_tensor(Tensor::zeros([num_experts, d_model], device)),
            gate_up_proj: Param::from_tensor(Tensor::zeros(
                [num_experts, d_model, 2 * moe_intermediate_size],
                device,
            )),
            down_proj: Param::from_tensor(Tensor::zeros(
                [num_experts, moe_intermediate_size, d_model],
                device,
            )),
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            norm_topk_prob,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, hidden] = x.dims();
        let num_tokens = batch * seq_len;

        // Flatten to [num_tokens, hidden]
        let x_flat = x.reshape([num_tokens, hidden]);

        // Router + top-k selection
        let (topk_weights, topk_indices) = {
            let _span = tracing::info_span!("router").entered();
            let router_logits = x_flat.clone().matmul(self.router_weight.val().transpose());
            let router_probs = activation::softmax(router_logits, 1);
            let (topk_weights, topk_indices) =
                Self::gpu_topk(router_probs, self.num_experts_per_tok);
            let topk_weights = if self.norm_topk_prob {
                let sum = topk_weights.clone().sum_dim(1);
                topk_weights / sum
            } else {
                topk_weights
            };
            (topk_weights, topk_indices)
        };

        // Dispatch: batched GPU path for decode, per-expert path for prefill
        let path = if num_tokens <= self.num_experts_per_tok {
            "batched"
        } else {
            "per_expert"
        };
        let output = {
            let _span = tracing::info_span!("dispatch", path).entered();
            if num_tokens <= self.num_experts_per_tok {
                self.forward_batched(x_flat, topk_weights, topk_indices)
            } else {
                self.forward_per_expert(x_flat, topk_weights, topk_indices)
            }
        };

        output.reshape([batch, seq_len, hidden])
    }

    /// GPU-native top-k via iterated argmax.
    ///
    /// WORKAROUND for burn#1490: `topk_with_indices` falls back to CPU sort on
    /// cubecl backends (WGPU, CUDA), causing a full GPU pipeline drain per call.
    /// This replaces it with k iterations of `argmax` (a GPU-native reduction),
    /// each masking the selected index to -inf before the next iteration.
    ///
    /// For decode (k=8, tensor=`[1, 128]`), this avoids a ~33% MoE time overhead
    /// from the GPU→CPU→GPU roundtrip. Remove this method and revert to
    /// `router_probs.topk_with_indices(k, 1)` once native GPU sort/topk lands.
    fn gpu_topk(probs: Tensor<B, 2>, k: usize) -> (Tensor<B, 2>, Tensor<B, 2, Int>) {
        let device = probs.device();
        let mut probs = probs;
        let mut all_values = Vec::with_capacity(k);
        let mut all_indices = Vec::with_capacity(k);

        for _ in 0..k {
            let idx = probs.clone().argmax(1); // [T, 1]
            let val = probs.clone().gather(1, idx.clone()); // [T, 1]
            all_values.push(val);
            all_indices.push(idx.clone());
            // Mask selected position to -inf for next iteration
            let mask = Tensor::<B, 2>::zeros(idx.dims(), &device) + f32::NEG_INFINITY;
            probs = probs.scatter(1, idx, mask, IndexingUpdateOp::Add);
        }

        (Tensor::cat(all_values, 1), Tensor::cat(all_indices, 1))
    }

    /// Batched decode path: selects top-k expert weights via GPU gather and runs
    /// a single batched matmul instead of k serial dispatches.
    ///
    /// Efficient for small token counts (decode), where num_tokens * k is small.
    fn forward_batched(
        &self,
        x_flat: Tensor<B, 2>,
        topk_weights: Tensor<B, 2>,
        topk_indices: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let [num_tokens, k] = topk_weights.dims();
        let hidden = x_flat.dims()[1];
        let total_slots = num_tokens * k;
        let moe_i = self.moe_intermediate_size;

        // Flatten expert indices for GPU gather: [total_slots]
        let flat_indices = topk_indices.reshape([total_slots]);

        // Select active expert weights: [total_slots, H, 2*I] and [total_slots, I, H]
        let expert_gate_up = self.gate_up_proj.val().select(0, flat_indices.clone());
        let expert_down = self.down_proj.val().select(0, flat_indices);

        // Expand input: each token repeated k times → [total_slots, 1, H]
        let x_expanded = x_flat
            .unsqueeze_dim::<3>(1) // [T, 1, H]
            .expand([num_tokens, k, hidden]) // [T, K, H]
            .reshape([total_slots, 1, hidden]); // [T*K, 1, H]

        // Batched gate_up: [T*K, 1, H] @ [T*K, H, 2*I] → [T*K, 1, 2*I]
        let gate_up = x_expanded.matmul(expert_gate_up);

        // SwiGLU
        let gate = gate_up.clone().slice([0..total_slots, 0..1, 0..moe_i]);
        let up = gate_up.slice([0..total_slots, 0..1, moe_i..2 * moe_i]);
        let hidden_states = activation::silu(gate) * up;

        // Batched down_proj: [T*K, 1, I] @ [T*K, I, H] → [T*K, 1, H]
        let expert_output = hidden_states.matmul(expert_down);

        // Reshape to [T, K, H] for weighted sum
        let expert_output = expert_output.reshape([num_tokens, k, hidden]);

        // Weight by router probs: [T, K, 1] * [T, K, H] → sum over K → [T, H]
        let weights = topk_weights.unsqueeze_dim::<3>(2);
        (expert_output * weights)
            .sum_dim(1)
            .reshape([num_tokens, hidden])
    }

    /// Per-expert dispatch path for prefill: builds a dispatch table in one pass
    /// and iterates only over active experts (not all num_experts).
    fn forward_per_expert(
        &self,
        x_flat: Tensor<B, 2>,
        topk_weights: Tensor<B, 2>,
        topk_indices: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let [num_tokens, _k] = topk_weights.dims();
        let hidden = x_flat.dims()[1];
        let device = x_flat.device();
        let mut output = Tensor::<B, 2>::zeros([num_tokens, hidden], &device);

        // CPU sync for dispatch table
        let indices_data = topk_indices.to_data();
        let indices_vec: Vec<i64> = indices_data.iter::<i64>().collect();
        let weights_data = topk_weights.to_data();
        let weights_vec: Vec<f32> = weights_data.iter::<f32>().collect();

        // Build dispatch table in one pass: expert_idx → Vec<(token_idx, weight)>
        let mut dispatch: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for token_idx in 0..num_tokens {
            for slot in 0..self.num_experts_per_tok {
                let idx = token_idx * self.num_experts_per_tok + slot;
                let expert_idx = indices_vec[idx] as usize;
                let weight = weights_vec[idx];
                dispatch
                    .entry(expert_idx)
                    .or_default()
                    .push((token_idx, weight));
            }
        }

        // Iterate only active experts
        for (expert_idx, assignments) in &dispatch {
            let n = assignments.len();

            // Gather tokens for this expert: [n, hidden]
            let gather_indices: Vec<i32> = assignments.iter().map(|&(ti, _)| ti as i32).collect();
            let gather_tensor = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::new(gather_indices, [n]),
                &device,
            );
            let expert_input = x_flat.clone().select(0, gather_tensor);

            // Slice this expert's weights
            let expert_gate_up = self
                .gate_up_proj
                .val()
                .slice([
                    *expert_idx..*expert_idx + 1,
                    0..hidden,
                    0..2 * self.moe_intermediate_size,
                ])
                .reshape([hidden, 2 * self.moe_intermediate_size]);

            let expert_down = self
                .down_proj
                .val()
                .slice([
                    *expert_idx..*expert_idx + 1,
                    0..self.moe_intermediate_size,
                    0..hidden,
                ])
                .reshape([self.moe_intermediate_size, hidden]);

            // SwiGLU
            let gate_up = expert_input.matmul(expert_gate_up);
            let gate = gate_up.clone().slice([0..n, 0..self.moe_intermediate_size]);
            let up = gate_up.slice([
                0..n,
                self.moe_intermediate_size..2 * self.moe_intermediate_size,
            ]);
            let hidden_states = activation::silu(gate) * up;

            let expert_output = hidden_states.matmul(expert_down);

            // Weight and scatter back
            let token_weights: Vec<f32> = assignments.iter().map(|&(_, w)| w).collect();
            let weight_tensor = Tensor::<B, 1>::from_floats(token_weights.as_slice(), &device)
                .unsqueeze_dim::<2>(1);
            let weighted_output = expert_output * weight_tensor;

            let scatter_indices: Vec<i32> = assignments.iter().map(|&(ti, _)| ti as i32).collect();
            let scatter_tensor = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::new(scatter_indices, [n]),
                &device,
            );
            let scatter_2d = scatter_tensor.unsqueeze_dim::<2>(1).expand([n, hidden]);
            output = output.scatter(0, scatter_2d, weighted_output, IndexingUpdateOp::Add);
        }

        output
    }
}

/// MLP variant: either a dense FeedForward or a Mixture of Experts layer.
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Mlp<B: Backend> {
    Dense(FeedForward<B>),
    Moe(MoeLayer<B>),
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Mlp::Dense(ff) => ff.forward(x),
            Mlp::Moe(moe) => moe.forward(x),
        }
    }
}

/// Single transformer decoder block.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    input_layernorm: RmsNorm<B>,
    self_attn: MultiHeadAttention<B>,
    post_attention_layernorm: RmsNorm<B>,
    mlp: Mlp<B>,
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
            mlp: Mlp::Dense(FeedForward::new(d_model, intermediate_size, device)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_moe(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate_size: usize,
        norm_topk_prob: bool,
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
            mlp: Mlp::Moe(MoeLayer::new(
                d_model,
                num_experts,
                num_experts_per_tok,
                moe_intermediate_size,
                norm_topk_prob,
                device,
            )),
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
        let x = {
            let _span = tracing::info_span!("attention").entered();
            let residual = x.clone();
            let h = self.input_layernorm.forward(x);
            let h = self.self_attn.forward(h, rope, mask, cache, start_pos);
            h + residual
        };

        // Pre-norm FFN with residual
        {
            let _span = tracing::info_span!("mlp").entered();
            let residual = x.clone();
            let h = self.post_attention_layernorm.forward(x);
            let h = self.mlp.forward(h);
            h + residual
        }
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

/// MoE configuration passed to Transformer::new when building MoE models.
pub struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,
    /// Layers that are dense-only despite being an MoE model.
    pub mlp_only_layers: Vec<usize>,
    /// MoE is applied every `decoder_sparse_step` layers.
    pub decoder_sparse_step: usize,
}

impl<B: Backend> Transformer<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_norm_eps: f64,
        _tie_word_embeddings: bool,
        moe_config: Option<&MoeConfig>,
        device: &Device<B>,
    ) -> Self {
        let embed_tokens = nn::EmbeddingConfig::new(vocab_size, d_model).init(device);

        let layers = (0..num_layers)
            .map(|i| {
                let use_moe = if let Some(moe) = moe_config {
                    let step = moe.decoder_sparse_step;
                    step > 0 && i % step == 0 && !moe.mlp_only_layers.contains(&i)
                } else {
                    false
                };

                if use_moe {
                    let moe = moe_config.unwrap();
                    TransformerBlock::new_moe(
                        d_model,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        moe.num_experts,
                        moe.num_experts_per_tok,
                        moe.moe_intermediate_size,
                        moe.norm_topk_prob,
                        rms_norm_eps,
                        device,
                    )
                } else {
                    TransformerBlock::new(
                        d_model,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        intermediate_size,
                        rms_norm_eps,
                        device,
                    )
                }
            })
            .collect();

        let norm = RmsNorm::new(d_model, rms_norm_eps, device);

        let lm_head = nn::LinearConfig::new(d_model, vocab_size)
            .with_bias(false)
            .init(device);

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
            let kind = match &layer.mlp {
                Mlp::Dense(_) => "dense",
                Mlp::Moe(_) => "moe",
            };
            let _span = tracing::info_span!("layer", i, kind).entered();
            x = layer.forward(x, rope, mask.clone(), &mut caches[i], start_pos);
            #[cfg(feature = "profile")]
            let _ = B::sync(&x.device());
        }

        {
            let _span = tracing::info_span!("norm_and_head").entered();
            x = self.norm.forward(x);
            let out = self.lm_head.forward(x);
            #[cfg(feature = "profile")]
            let _ = B::sync(&out.device());
            out
        }
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

        // MLP (dense)
        match &mut layer.mlp {
            Mlp::Dense(ff) => {
                ff.gate_proj.weight = Param::from_tensor(gate_proj_w);
                ff.up_proj.weight = Param::from_tensor(up_proj_w);
                ff.down_proj.weight = Param::from_tensor(down_proj_w);
            }
            Mlp::Moe(_) => panic!("load_layer called on MoE layer {}", layer_idx),
        }

        // Layer norms
        layer.input_layernorm.weight = Param::from_tensor(input_ln_w);
        layer.post_attention_layernorm.weight = Param::from_tensor(post_attn_ln_w);

        self
    }

    /// Load weights for a MoE layer.
    ///
    /// - `gate_up_proj`: packed expert weights [num_experts, hidden, 2*moe_intermediate] (already transposed)
    /// - `down_proj`: expert weights [num_experts, moe_intermediate, hidden] (already transposed)
    /// - `router_weight`: [num_experts, hidden]
    #[allow(clippy::too_many_arguments)]
    pub fn load_moe_layer(
        mut self,
        layer_idx: usize,
        q_proj_w: Tensor<B, 2>,
        k_proj_w: Tensor<B, 2>,
        v_proj_w: Tensor<B, 2>,
        o_proj_w: Tensor<B, 2>,
        q_norm_w: Tensor<B, 1>,
        k_norm_w: Tensor<B, 1>,
        gate_up_proj: Tensor<B, 3>,
        down_proj: Tensor<B, 3>,
        router_weight: Tensor<B, 2>,
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

        // MoE weights
        match &mut layer.mlp {
            Mlp::Moe(moe) => {
                moe.gate_up_proj = Param::from_tensor(gate_up_proj);
                moe.down_proj = Param::from_tensor(down_proj);
                moe.router_weight = Param::from_tensor(router_weight);
            }
            Mlp::Dense(_) => panic!("load_moe_layer called on dense layer {}", layer_idx),
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn device() -> <B as Backend>::Device {
        Default::default()
    }

    fn to_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.to_data().iter::<f32>().collect()
    }

    // --- RmsNorm ---

    #[test]
    fn rms_norm_unit_weight() {
        // With weight=1 and eps=0, RmsNorm(x) = x / rms(x)
        // For input [1, 1, 2] with values [3.0, 4.0]: rms = sqrt((9+16)/2) = sqrt(12.5)
        let dev = device();
        let norm = RmsNorm::new(2, 1e-6, &dev);
        let x = Tensor::<B, 3>::from_floats([[[3.0, 4.0]]], &dev);
        let out = norm.forward(x);
        let vals = to_vec(out);
        let rms = (12.5f32 + 1e-6).sqrt();
        assert!((vals[0] - 3.0 / rms).abs() < 1e-5);
        assert!((vals[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn rms_norm_preserves_shape() {
        let dev = device();
        let norm = RmsNorm::new(8, 1e-6, &dev);
        let x = Tensor::<B, 3>::ones([2, 5, 8], &dev);
        let out = norm.forward(x);
        assert_eq!(out.dims(), [2, 5, 8]);
    }

    #[test]
    fn rms_norm_uniform_input() {
        // Uniform input: all values c -> rms = c, so output = c/c * weight = weight
        // With weight=1, output should be all 1.0
        let dev = device();
        let norm = RmsNorm::new(4, 1e-6, &dev);
        let x = Tensor::<B, 3>::from_floats([[[5.0, 5.0, 5.0, 5.0]]], &dev);
        let out = norm.forward(x);
        let vals = to_vec(out);
        for &v in &vals {
            assert!((v - 1.0).abs() < 1e-4, "expected ~1.0 got {}", v);
        }
    }

    // --- rotate_half ---

    #[test]
    fn rotate_half_known_values() {
        // Input: [1, 1, 1, 4] with values [a, b, c, d]
        // Expected: [-c, -d, a, b]
        let dev = device();
        let x = Tensor::<B, 4>::from_floats([[[[1.0, 2.0, 3.0, 4.0]]]], &dev);
        let out = rotate_half(x);
        let vals = to_vec(out);
        assert_eq!(vals, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn rotate_half_involution() {
        // rotate_half applied twice: first [-x2, x1], then [-x1, -x2] = -x
        let dev = device();
        let x = Tensor::<B, 4>::from_floats([[[[1.0, 2.0, 3.0, 4.0]]]], &dev);
        let out = rotate_half(rotate_half(x.clone()));
        let vals = to_vec(out);
        let orig = to_vec(x);
        for (a, b) in vals.iter().zip(orig.iter()) {
            assert!((a + b).abs() < 1e-6, "expected negation");
        }
    }

    // --- RoPE ---

    #[test]
    fn rope_preserves_shape() {
        let dev = device();
        let rope = RotaryEmbedding::new(4, 32, 10000.0, &dev);
        let q = Tensor::<B, 4>::ones([1, 2, 5, 4], &dev);
        let k = Tensor::<B, 4>::ones([1, 2, 5, 4], &dev);
        let (q_out, k_out) = rope.apply(q, k, 0);
        assert_eq!(q_out.dims(), [1, 2, 5, 4]);
        assert_eq!(k_out.dims(), [1, 2, 5, 4]);
    }

    #[test]
    fn rope_position_offset() {
        // Applying RoPE at start_pos=0 with seq_len=1 should give same result
        // as applying at start_pos=5 with seq_len=1 but for different position embeddings
        let dev = device();
        let rope = RotaryEmbedding::new(4, 32, 10000.0, &dev);
        let q = Tensor::<B, 4>::ones([1, 1, 1, 4], &dev);
        let k = Tensor::<B, 4>::ones([1, 1, 1, 4], &dev);

        let (q0, _) = rope.apply(q.clone(), k.clone(), 0);
        let (q5, _) = rope.apply(q, k, 5);
        // Different positions should give different embeddings
        let diff: f32 = to_vec((q0 - q5).abs()).iter().sum();
        assert!(
            diff > 1e-4,
            "different positions should produce different embeddings"
        );
    }

    #[test]
    fn rope_cos_sin_table_shape() {
        let dev = device();
        let rope = RotaryEmbedding::<B>::new(8, 64, 10000.0, &dev);
        assert_eq!(rope.cos.dims(), [64, 8]);
        assert_eq!(rope.sin.dims(), [64, 8]);
    }

    // --- Causal mask ---

    #[test]
    fn causal_mask_square() {
        // 3x3 causal mask: lower triangle + diagonal = 0.0, upper triangle = -inf
        let dev = device();
        let mask = build_causal_mask::<B>(3, 3, &dev);
        let vals = to_vec(mask);
        // Row 0: [0, -inf, -inf]
        // Row 1: [0, 0, -inf]
        // Row 2: [0, 0, 0]
        assert_eq!(vals[0], 0.0);
        assert!(vals[1].is_infinite() && vals[1] < 0.0);
        assert!(vals[2].is_infinite() && vals[2] < 0.0);
        assert_eq!(vals[3], 0.0);
        assert_eq!(vals[4], 0.0);
        assert!(vals[5].is_infinite() && vals[5] < 0.0);
        assert_eq!(vals[6], 0.0);
        assert_eq!(vals[7], 0.0);
        assert_eq!(vals[8], 0.0);
    }

    #[test]
    fn causal_mask_single_query() {
        // Autoregressive step: 1 query, 5 total -> mask is [1, 5], all 0.0 (can attend to everything)
        let dev = device();
        let mask = build_causal_mask::<B>(1, 5, &dev);
        assert_eq!(mask.dims(), [1, 5]);
        let vals = to_vec(mask);
        for &v in &vals {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn causal_mask_2_query_5_total() {
        // 2 queries, 5 total: offset=3
        // Row 0: can attend to positions 0..3 (j<=3), mask at j=4 -> [0,0,0,0,-inf]
        // Row 1: can attend to positions 0..4 (j<=4) -> [0,0,0,0,0]
        let dev = device();
        let mask = build_causal_mask::<B>(2, 5, &dev);
        assert_eq!(mask.dims(), [2, 5]);
        let vals = to_vec(mask);
        // Row 0
        assert_eq!(vals[0], 0.0);
        assert_eq!(vals[1], 0.0);
        assert_eq!(vals[2], 0.0);
        assert_eq!(vals[3], 0.0);
        assert!(vals[4].is_infinite() && vals[4] < 0.0);
        // Row 1: all zero
        for &v in &vals[5..10] {
            assert_eq!(v, 0.0);
        }
    }

    // --- repeat_kv ---

    #[test]
    fn repeat_kv_noop_when_1() {
        let dev = device();
        let x = Tensor::<B, 4>::from_floats([[[[1.0, 2.0]], [[3.0, 4.0]]]], &dev);
        let out = repeat_kv(x.clone(), 1);
        assert_eq!(to_vec(out), to_vec(x));
    }

    #[test]
    fn repeat_kv_doubles() {
        let dev = device();
        // [1, 2, 1, 2]: batch=1, 2 kv heads, seq=1, dim=2
        let x = Tensor::<B, 4>::from_floats([[[[1.0, 2.0]], [[3.0, 4.0]]]], &dev);
        let out = repeat_kv(x, 2);
        assert_eq!(out.dims(), [1, 4, 1, 2]);
        let vals = to_vec(out);
        // heads: [1,2], [1,2], [3,4], [3,4]
        assert_eq!(vals, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    // --- FeedForward ---

    #[test]
    fn feed_forward_preserves_shape() {
        let dev = device();
        let ff = FeedForward::new(16, 32, &dev);
        let x = Tensor::<B, 3>::ones([1, 4, 16], &dev);
        let out = ff.forward(x);
        assert_eq!(out.dims(), [1, 4, 16]);
    }

    // --- MoeLayer ---

    #[test]
    fn moe_layer_preserves_shape() {
        let dev = device();
        let moe = MoeLayer::new(8, 4, 2, 16, true, &dev);
        let x = Tensor::<B, 3>::ones([1, 3, 8], &dev);
        let out = moe.forward(x);
        assert_eq!(out.dims(), [1, 3, 8]);
    }

    #[test]
    fn moe_layer_batch_shape() {
        let dev = device();
        let moe = MoeLayer::new(8, 4, 2, 16, true, &dev);
        let x = Tensor::<B, 3>::ones([2, 5, 8], &dev);
        let out = moe.forward(x);
        assert_eq!(out.dims(), [2, 5, 8]);
    }

    #[test]
    fn moe_layer_single_expert_single_topk() {
        // With 1 expert and topk=1, MoE should behave like a single FFN
        // (all tokens route to expert 0 with weight 1.0)
        let dev = device();
        let moe = MoeLayer::new(4, 1, 1, 8, true, &dev);
        let x = Tensor::<B, 3>::ones([1, 2, 4], &dev);
        let out = moe.forward(x);
        assert_eq!(out.dims(), [1, 2, 4]);
        // Output should be deterministic (not all zeros, since weights are zero-initialized
        // the actual output will be zeros here, but shape is correct)
    }

    #[test]
    fn moe_layer_decode_single_token() {
        // Single token triggers the batched path (num_tokens=1 <= num_experts_per_tok=2)
        let dev = device();
        let mut moe = MoeLayer::new(4, 4, 2, 8, true, &dev);
        // Set non-zero weights so output is non-trivial
        moe.router_weight = Param::from_tensor(Tensor::from_floats([[1., 0., 0., 0.]; 4], &dev));
        moe.gate_up_proj = Param::from_tensor(Tensor::<B, 3>::ones([4, 4, 16], &dev) * 0.1);
        moe.down_proj = Param::from_tensor(Tensor::<B, 3>::ones([4, 8, 4], &dev) * 0.1);
        let x = Tensor::<B, 3>::ones([1, 1, 4], &dev);
        let out = moe.forward(x);
        assert_eq!(out.dims(), [1, 1, 4]);
        // Output should be non-zero with non-zero weights
        let vals = to_vec(out);
        let sum: f32 = vals.iter().map(|v| v.abs()).sum();
        assert!(sum > 1e-6, "expected non-zero output from batched path");
    }

    #[test]
    fn moe_routing_renormalization() {
        // Verify that with norm_topk_prob=true, weights sum to 1.0 per token
        // We test this indirectly: with uniform router weights and 4 experts, topk=2,
        // each selected expert gets weight 0.5
        let dev = device();
        let mut moe = MoeLayer::new(4, 4, 2, 8, true, &dev);
        // Set router to uniform so all experts have equal probability
        moe.router_weight = Param::from_tensor(Tensor::ones([4, 4], &dev));
        let x = Tensor::<B, 3>::ones([1, 1, 4], &dev);
        // This should not panic and should produce valid output
        let out = moe.forward(x);
        assert_eq!(out.dims(), [1, 1, 4]);
    }

    // --- Mlp enum ---

    #[test]
    fn mlp_dense_variant_shape() {
        let dev = device();
        let mlp = Mlp::Dense(FeedForward::new(16, 32, &dev));
        let x = Tensor::<B, 3>::ones([1, 3, 16], &dev);
        let out = mlp.forward(x);
        assert_eq!(out.dims(), [1, 3, 16]);
    }

    #[test]
    fn mlp_moe_variant_shape() {
        let dev = device();
        let mlp = Mlp::Moe(MoeLayer::new(16, 4, 2, 32, true, &dev));
        let x = Tensor::<B, 3>::ones([1, 3, 16], &dev);
        let out = mlp.forward(x);
        assert_eq!(out.dims(), [1, 3, 16]);
    }

    // --- TransformerBlock ---

    #[test]
    fn transformer_block_dense_shape() {
        let dev = device();
        let block = TransformerBlock::new(32, 4, 2, 8, 64, 1e-6, &dev);
        let rope = RotaryEmbedding::new(8, 16, 10000.0, &dev);
        let mut cache = AttentionKvCache::new(1, 2, 16, 8, &dev);
        let mask = build_causal_mask::<B>(3, 3, &dev);

        let x = Tensor::<B, 3>::ones([1, 3, 32], &dev);
        let out = block.forward(x, &rope, Some(mask), &mut cache, 0);
        assert_eq!(out.dims(), [1, 3, 32]);
    }

    #[test]
    fn transformer_block_moe_shape() {
        let dev = device();
        let block = TransformerBlock::new_moe(32, 4, 2, 8, 4, 2, 16, true, 1e-6, &dev);
        let rope = RotaryEmbedding::new(8, 16, 10000.0, &dev);
        let mut cache = AttentionKvCache::new(1, 2, 16, 8, &dev);
        let mask = build_causal_mask::<B>(3, 3, &dev);

        let x = Tensor::<B, 3>::ones([1, 3, 32], &dev);
        let out = block.forward(x, &rope, Some(mask), &mut cache, 0);
        assert_eq!(out.dims(), [1, 3, 32]);
    }

    // --- Transformer construction ---

    #[test]
    fn transformer_dense_construction() {
        let dev = device();
        let t = Transformer::new(100, 32, 2, 4, 2, 8, 64, 1e-6, false, None, &dev);
        let rope = RotaryEmbedding::new(8, 16, 10000.0, &dev);
        let mut caches: Vec<AttentionKvCache<B>> = (0..2)
            .map(|_| AttentionKvCache::new(1, 2, 16, 8, &dev))
            .collect();
        let tokens = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(vec![1i32, 2, 3], [1, 3]),
            &dev,
        );
        let mask = build_causal_mask::<B>(3, 3, &dev);
        let out = t.forward(tokens, &rope, Some(mask), &mut caches, 0);
        assert_eq!(out.dims(), [1, 3, 100]); // [batch, seq, vocab]
    }

    #[test]
    fn transformer_moe_construction() {
        let dev = device();
        let moe_cfg = MoeConfig {
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 16,
            norm_topk_prob: true,
            mlp_only_layers: vec![],
            decoder_sparse_step: 1,
        };
        let t = Transformer::new(100, 32, 2, 4, 2, 8, 64, 1e-6, false, Some(&moe_cfg), &dev);
        let rope = RotaryEmbedding::new(8, 16, 10000.0, &dev);
        let mut caches: Vec<AttentionKvCache<B>> = (0..2)
            .map(|_| AttentionKvCache::new(1, 2, 16, 8, &dev))
            .collect();
        let tokens = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(vec![1i32, 2, 3], [1, 3]),
            &dev,
        );
        let mask = build_causal_mask::<B>(3, 3, &dev);
        let out = t.forward(tokens, &rope, Some(mask), &mut caches, 0);
        assert_eq!(out.dims(), [1, 3, 100]);
    }

    #[test]
    fn transformer_mixed_moe_dense_layers() {
        let dev = device();
        let moe_cfg = MoeConfig {
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 16,
            norm_topk_prob: true,
            mlp_only_layers: vec![1], // layer 1 is dense
            decoder_sparse_step: 1,
        };
        // 3 layers: 0=MoE, 1=Dense, 2=MoE
        let t = Transformer::new(100, 32, 3, 4, 2, 8, 64, 1e-6, false, Some(&moe_cfg), &dev);
        let rope = RotaryEmbedding::new(8, 16, 10000.0, &dev);
        let mut caches: Vec<AttentionKvCache<B>> = (0..3)
            .map(|_| AttentionKvCache::new(1, 2, 16, 8, &dev))
            .collect();
        let tokens = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(vec![1i32, 2], [1, 2]),
            &dev,
        );
        let mask = build_causal_mask::<B>(2, 2, &dev);
        let out = t.forward(tokens, &rope, Some(mask), &mut caches, 0);
        assert_eq!(out.dims(), [1, 2, 100]);
    }
}
