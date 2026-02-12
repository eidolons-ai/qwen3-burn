use burn::module::Param;
use burn::prelude::*;
use burn::tensor::activation;

use serde::Deserialize;

/// Vision encoder configuration (from `vision_config` in config.json).
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_vision_hidden")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_out_hidden")]
    pub out_hidden_size: usize,
    #[serde(default = "default_vision_depth")]
    pub depth: usize,
    #[serde(default = "default_vision_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_vision_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_spatial_merge")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_vision_temporal_patch")]
    pub temporal_patch_size: usize,
    #[serde(default)]
    pub deepstack_visual_indexes: Option<Vec<usize>>,
    #[serde(default = "default_vision_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_vision_num_pos_embed")]
    pub num_position_embeddings: usize,
}

fn default_vision_hidden() -> usize {
    1024
}
fn default_vision_out_hidden() -> usize {
    2048
}
fn default_vision_depth() -> usize {
    24
}
fn default_vision_num_heads() -> usize {
    16
}
fn default_vision_intermediate() -> usize {
    4096
}
fn default_vision_patch_size() -> usize {
    16
}
fn default_vision_spatial_merge() -> usize {
    2
}
fn default_vision_temporal_patch() -> usize {
    2
}
fn default_vision_in_channels() -> usize {
    3
}
fn default_vision_num_pos_embed() -> usize {
    2304
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            out_hidden_size: 2048,
            depth: 24,
            num_heads: 16,
            intermediate_size: 4096,
            patch_size: 16,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            deepstack_visual_indexes: Some(vec![5, 11, 17]),
            in_channels: 3,
            num_position_embeddings: 2304,
        }
    }
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Patch embed input dimension: in_channels * temporal_patch_size * patch_size * patch_size
    pub fn patch_embed_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// The merged hidden size after spatial 2x2 merge.
    pub fn merged_hidden_size(&self) -> usize {
        self.hidden_size * self.spatial_merge_size * self.spatial_merge_size
    }

    /// Qwen3-VL-2B vision config.
    pub fn qwen3_vl_2b() -> Self {
        Self::default()
    }
}

// --- LayerNorm ---

/// Standard Layer Normalization with weight and bias.
#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(size: usize, eps: f64, device: &Device<B>) -> Self {
        let weight = Tensor::ones([size], device);
        let bias = Tensor::zeros([size], device);
        Self {
            weight: Param::from_tensor(weight),
            bias: Param::from_tensor(bias),
            eps,
        }
    }

    /// Forward on 2D input `[seq, hidden]`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mean = x.clone().mean_dim(1);
        let diff = x - mean;
        let variance = diff.clone().powf_scalar(2.0).mean_dim(1);
        let normed = diff * (variance + self.eps).sqrt().recip();
        normed * self.weight.val().unsqueeze_dim::<2>(0) + self.bias.val().unsqueeze_dim::<2>(0)
    }

    /// Forward on 3D input `[batch, seq, hidden]`.
    pub fn forward_3d(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = x.clone().mean_dim(2);
        let diff = x - mean;
        let variance = diff.clone().powf_scalar(2.0).mean_dim(2);
        let normed = diff * (variance + self.eps).sqrt().recip();
        normed * self.weight.val().unsqueeze::<3>() + self.bias.val().unsqueeze::<3>()
    }
}

// --- GELU with tanh approximation ---

/// GELU activation with tanh approximation:
/// `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
pub fn gelu_tanh<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    // Use Burn's built-in Gelu which uses tanh approximation
    activation::gelu(x)
}

// --- Vision Rotary Embedding ---

/// 2D spatial RoPE for vision encoder (theta=10000).
///
/// Matches HuggingFace's VisionRotaryEmbedding:
/// - `dim = head_dim / 2` (e.g., 32 for head_dim=64)
/// - 16 inverse frequencies: `1 / (theta ^ (arange(0, dim, 2) / dim))`
/// - Each patch gets (row_freq, col_freq) pairs, duplicated to head_dim
pub struct VisionRotaryEmbedding<B: Backend> {
    /// 16 inverse frequencies (dim/2 values for dim = head_dim/2)
    inv_freq: Vec<f64>,
    head_dim: usize,
    _device: Device<B>,
}

impl<B: Backend> VisionRotaryEmbedding<B> {
    pub fn new(head_dim: usize, _max_seq_len: usize, device: &Device<B>) -> Self {
        let theta: f64 = 10000.0;
        // Match HF: dim = head_dim / 2, then arange(0, dim, 2) gives dim/2 frequencies
        let dim = head_dim / 2; // 32 for head_dim=64
        let num_freqs = dim / 2; // 16

        let inv_freq: Vec<f64> = (0..num_freqs)
            .map(|i| 1.0 / theta.powf(i as f64 * 2.0 / dim as f64))
            .collect();

        Self {
            inv_freq,
            head_dim,
            _device: device.clone(),
        }
    }

    /// Compute cos/sin for given 1D position IDs `[seq_len]`.
    /// Returns `(cos, sin)` each of shape `[seq_len, head_dim]`.
    /// Note: For 2D vision positions, use VisionEncoder::compute_vision_rope_2d instead.
    pub fn forward(&self, position_ids: &[usize]) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let seq_len = position_ids.len();
        let num_freqs = self.inv_freq.len(); // 16
        let mut cos_data = Vec::with_capacity(seq_len * self.head_dim);
        let mut sin_data = Vec::with_capacity(seq_len * self.head_dim);

        for &pos in position_ids {
            // Compute 16 angle values
            for i in 0..num_freqs {
                let angle = pos as f64 * self.inv_freq[i];
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
            // Duplicate: 16 -> 32 -> 64 (head_dim)
            // First duplicate to 32
            let start = cos_data.len() - num_freqs;
            cos_data.extend_from_within(start..);
            sin_data.extend_from_within(start..);
            // Then duplicate 32 to 64
            let start = cos_data.len() - 2 * num_freqs;
            cos_data.extend_from_within(start..);
            sin_data.extend_from_within(start..);
        }

        let cos = Tensor::<B, 1>::from_floats(&cos_data[..], &self._device)
            .reshape([seq_len, self.head_dim]);
        let sin = Tensor::<B, 1>::from_floats(&sin_data[..], &self._device)
            .reshape([seq_len, self.head_dim]);

        (cos, sin)
    }
}

/// Rotate the second half of the last dimension for 3D tensors: [-x2, x1]
fn rotate_half_3d<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [a, b, dim] = x.dims();
    let half = dim / 2;
    let x1 = x.clone().slice([0..a, 0..b, 0..half]);
    let x2 = x.slice([0..a, 0..b, half..dim]);
    Tensor::cat(vec![x2.neg(), x1], 2)
}

/// Apply rotary embeddings to 3D tensors `[batch*heads, seq_len, head_dim]`.
fn apply_vision_rope<B: Backend>(
    x: Tensor<B, 3>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> Tensor<B, 3> {
    // cos/sin: [seq_len, head_dim] -> unsqueeze to [1, seq_len, head_dim]
    let cos = cos.unsqueeze_dim::<3>(0);
    let sin = sin.unsqueeze_dim::<3>(0);
    x.clone() * cos + rotate_half_3d(x) * sin
}

// --- VisionAttention ---

/// Full self-attention for vision encoder with fused QKV and bias.
#[derive(Module, Debug)]
pub struct VisionAttention<B: Backend> {
    qkv: nn::Linear<B>,
    proj: nn::Linear<B>,
    num_heads: usize,
    head_dim: usize,
}

impl<B: Backend> VisionAttention<B> {
    pub fn new(hidden_size: usize, num_heads: usize, device: &Device<B>) -> Self {
        let head_dim = hidden_size / num_heads;
        Self {
            qkv: nn::LinearConfig::new(hidden_size, 3 * hidden_size)
                .with_bias(true)
                .init(device),
            proj: nn::LinearConfig::new(hidden_size, hidden_size)
                .with_bias(true)
                .init(device),
            num_heads,
            head_dim,
        }
    }

    /// Forward pass with per-sequence attention.
    ///
    /// - `x`: `[total_seq, hidden]`
    /// - `cu_seqlens`: cumulative sequence lengths (e.g., `[0, 100, 200]` for two sequences)
    /// - `cos_sin`: optional pre-computed `(cos, sin)` each `[total_seq, head_dim]`
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        cu_seqlens: &[usize],
        cos_sin: Option<&(Tensor<B, 2>, Tensor<B, 2>)>,
    ) -> Tensor<B, 2> {
        let [total_seq, hidden] = x.dims();

        // Project QKV: [total_seq, 3*hidden]
        let qkv = self
            .qkv
            .forward(x.clone().unsqueeze_dim::<3>(0))
            .reshape([total_seq, 3 * hidden]);

        // Split into Q, K, V: each [total_seq, hidden]
        let q = qkv.clone().slice([0..total_seq, 0..hidden]);
        let k = qkv.clone().slice([0..total_seq, hidden..2 * hidden]);
        let v = qkv.slice([0..total_seq, 2 * hidden..3 * hidden]);

        // Reshape to [total_seq, num_heads, head_dim]
        let q = q.reshape([total_seq, self.num_heads, self.head_dim]);
        let k = k.reshape([total_seq, self.num_heads, self.head_dim]);
        let v = v.reshape([total_seq, self.num_heads, self.head_dim]);

        // Apply RoPE to all Q, K at once if provided
        let (q, k) = if let Some((cos, sin)) = cos_sin {
            // q: [total_seq, num_heads, head_dim] -> [num_heads, total_seq, head_dim]
            let q_r = q.swap_dims(0, 1);
            let k_r = k.swap_dims(0, 1);
            let q_r = apply_vision_rope(q_r, cos.clone(), sin.clone());
            let k_r = apply_vision_rope(k_r, cos.clone(), sin.clone());
            (q_r.swap_dims(0, 1), k_r.swap_dims(0, 1))
        } else {
            (q, k)
        };

        // Process per-sequence with attention boundaries
        let num_seqs = cu_seqlens.len() - 1;
        let mut outputs = Vec::with_capacity(num_seqs);

        for seq_idx in 0..num_seqs {
            let start = cu_seqlens[seq_idx];
            let end = cu_seqlens[seq_idx + 1];
            let seq_len = end - start;

            // Slice this sequence: [seq_len, num_heads, head_dim]
            let q_seq = q
                .clone()
                .slice([start..end, 0..self.num_heads, 0..self.head_dim]);
            let k_seq = k
                .clone()
                .slice([start..end, 0..self.num_heads, 0..self.head_dim]);
            let v_seq = v
                .clone()
                .slice([start..end, 0..self.num_heads, 0..self.head_dim]);

            // Transpose to [num_heads, seq_len, head_dim] for attention
            let q_h = q_seq.swap_dims(0, 1); // [num_heads, seq_len, head_dim]
            let k_h = k_seq.swap_dims(0, 1);
            let v_h = v_seq.swap_dims(0, 1);

            // Scaled dot-product attention (bidirectional, no mask)
            let scale = (self.head_dim as f64).sqrt().recip();
            let attn_weights = q_h.matmul(k_h.transpose()) * scale;
            let attn_weights = activation::softmax(attn_weights, 2);
            let attn_output = attn_weights.matmul(v_h); // [num_heads, seq_len, head_dim]

            // Transpose back to [seq_len, num_heads, head_dim] and reshape to [seq_len, hidden]
            let attn_output = attn_output.swap_dims(0, 1).reshape([seq_len, hidden]);
            outputs.push(attn_output);
        }

        // Concatenate all sequences: [total_seq, hidden]
        let output = Tensor::cat(outputs, 0);

        // Output projection
        self.proj
            .forward(output.unsqueeze_dim::<3>(0))
            .reshape([total_seq, hidden])
    }
}

// --- VisionMLP ---

/// Standard GELU MLP for vision encoder.
#[derive(Module, Debug)]
pub struct VisionMLP<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
}

impl<B: Backend> VisionMLP<B> {
    pub fn new(hidden_size: usize, intermediate_size: usize, device: &Device<B>) -> Self {
        Self {
            fc1: nn::LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(true)
                .init(device),
            fc2: nn::LinearConfig::new(intermediate_size, hidden_size)
                .with_bias(true)
                .init(device),
        }
    }

    /// Forward on 2D input `[seq, hidden]`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [seq, _] = x.dims();
        // Use 3D for nn::Linear
        let x = x.unsqueeze_dim::<3>(0);
        let h = gelu_tanh(self.fc1.forward(x));
        let out = self.fc2.forward(h);
        let out_dim = out.dims()[2];
        out.reshape([seq, out_dim])
    }
}

// --- VisionBlock ---

/// Vision encoder transformer block (pre-norm residual).
#[derive(Module, Debug)]
pub struct VisionBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attn: VisionAttention<B>,
    norm2: LayerNorm<B>,
    mlp: VisionMLP<B>,
}

impl<B: Backend> VisionBlock<B> {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        eps: f64,
        device: &Device<B>,
    ) -> Self {
        Self {
            norm1: LayerNorm::new(hidden_size, eps, device),
            attn: VisionAttention::new(hidden_size, num_heads, device),
            norm2: LayerNorm::new(hidden_size, eps, device),
            mlp: VisionMLP::new(hidden_size, intermediate_size, device),
        }
    }

    /// Forward pass.
    ///
    /// - `x`: `[total_seq, hidden]`
    /// - `cu_seqlens`: cumulative sequence lengths
    /// - `cos_sin`: optional pre-computed `(cos, sin)` for RoPE
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        cu_seqlens: &[usize],
        cos_sin: Option<&(Tensor<B, 2>, Tensor<B, 2>)>,
    ) -> Tensor<B, 2> {
        let residual = x.clone();
        let h = self.norm1.forward(x);
        let h = self.attn.forward(h, cu_seqlens, cos_sin);
        let x = h + residual;

        let residual = x.clone();
        let h = self.norm2.forward(x);
        let h = self.mlp.forward(h);
        h + residual
    }
}

// --- PatchMerger ---

/// Spatial 2x2 merge + MLP projection.
///
/// Two modes:
/// - `use_postshuffle_norm=false` (main merger): norm on hidden_size, then spatial merge, then MLP
/// - `use_postshuffle_norm=true` (deepstack mergers): spatial merge first, then norm on merged_size, then MLP
#[derive(Module, Debug)]
pub struct PatchMerger<B: Backend> {
    norm: LayerNorm<B>,
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    spatial_merge_size: usize,
    use_postshuffle_norm: bool,
}

impl<B: Backend> PatchMerger<B> {
    pub fn new(
        hidden_size: usize,
        out_hidden_size: usize,
        spatial_merge_size: usize,
        use_postshuffle_norm: bool,
        eps: f64,
        device: &Device<B>,
    ) -> Self {
        let merged_size = hidden_size * spatial_merge_size * spatial_merge_size;
        let norm_size = if use_postshuffle_norm {
            merged_size
        } else {
            hidden_size
        };
        Self {
            norm: LayerNorm::new(norm_size, eps, device),
            fc1: nn::LinearConfig::new(merged_size, merged_size)
                .with_bias(true)
                .init(device),
            fc2: nn::LinearConfig::new(merged_size, out_hidden_size)
                .with_bias(true)
                .init(device),
            spatial_merge_size,
            use_postshuffle_norm,
        }
    }

    /// Forward pass.
    ///
    /// - `x`: `[num_patches, hidden_size]`
    /// - `grid_h`, `grid_w`: spatial grid dimensions of the patches
    ///
    /// Returns `[num_merged_patches, out_hidden_size]` where `num_merged_patches = num_patches / merge_size^2`.
    pub fn forward(&self, x: Tensor<B, 2>, grid_h: usize, grid_w: usize) -> Tensor<B, 2> {
        let merge = self.spatial_merge_size;
        let merged_h = grid_h / merge;
        let merged_w = grid_w / merge;
        let device = x.device();

        if !self.use_postshuffle_norm {
            // Pre-shuffle norm: normalize on hidden_size first
            let x = self.norm.forward(x);
            // Then spatial merge
            let x = spatial_merge(x, grid_h, grid_w, merge, &device);
            // MLP
            let x = x.unsqueeze_dim::<3>(0);
            let x = gelu_tanh(self.fc1.forward(x));
            let x = self.fc2.forward(x);
            let out_dim = x.dims()[2];
            x.reshape([merged_h * merged_w, out_dim])
        } else {
            // Post-shuffle norm: spatial merge first
            let x = spatial_merge(x, grid_h, grid_w, merge, &device);
            // Then normalize on merged_size
            let x = self.norm.forward(x);
            // MLP
            let x = x.unsqueeze_dim::<3>(0);
            let x = gelu_tanh(self.fc1.forward(x));
            let x = self.fc2.forward(x);
            let out_dim = x.dims()[2];
            x.reshape([merged_h * merged_w, out_dim])
        }
    }
}

/// Merge adjacent spatial patches into groups.
///
/// Input: `[grid_h * grid_w, hidden]` patches in row-major order.
/// Output: `[merged_h * merged_w, hidden * merge^2]` where adjacent merge x merge patches are concatenated.
fn spatial_merge<B: Backend>(
    x: Tensor<B, 2>,
    grid_h: usize,
    grid_w: usize,
    merge: usize,
    device: &Device<B>,
) -> Tensor<B, 2> {
    let [_total, hidden] = x.dims();
    let merged_h = grid_h / merge;
    let merged_w = grid_w / merge;
    let merged_size = hidden * merge * merge;

    // Reshape: [grid_h, grid_w, hidden] -> [merged_h, merge, merged_w, merge, hidden]
    // -> [merged_h, merged_w, merge, merge, hidden] -> [merged_h*merged_w, merge*merge*hidden]
    // We'll do this with explicit index gathering since burn doesn't have arbitrary reshape/permute
    let x_data: Vec<f32> = x.to_data().iter::<f32>().collect();

    let mut output = vec![0.0f32; merged_h * merged_w * merged_size];
    for mh in 0..merged_h {
        for mw in 0..merged_w {
            let out_idx = mh * merged_w + mw;
            let mut offset = 0;
            for dh in 0..merge {
                for dw in 0..merge {
                    let src_h = mh * merge + dh;
                    let src_w = mw * merge + dw;
                    let src_idx = src_h * grid_w + src_w;
                    let src_start = src_idx * hidden;
                    output[out_idx * merged_size + offset..out_idx * merged_size + offset + hidden]
                        .copy_from_slice(&x_data[src_start..src_start + hidden]);
                    offset += hidden;
                }
            }
        }
    }

    Tensor::<B, 1>::from_floats(&output[..], device).reshape([merged_h * merged_w, merged_size])
}

/// Apply a merger to each temporal frame independently, then concatenate.
///
/// For `grid_t == 1` (images), passes through directly. For `grid_t > 1` (video),
/// slices each frame's patches, merges separately, and concatenates results.
fn merge_per_frame<B: Backend>(
    hidden_states: &Tensor<B, 2>,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    merge_fn: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
) -> Tensor<B, 2> {
    if grid_t == 1 {
        merge_fn(hidden_states.clone())
    } else {
        let patches_per_frame = grid_h * grid_w;
        let [_total, hidden] = hidden_states.dims();
        let frames: Vec<Tensor<B, 2>> = (0..grid_t)
            .map(|t| {
                let start = t * patches_per_frame;
                let end = start + patches_per_frame;
                let frame = hidden_states.clone().slice([start..end, 0..hidden]);
                merge_fn(frame)
            })
            .collect();
        Tensor::cat(frames, 0)
    }
}

// --- VisionEncoder ---

/// Full SigLIP-2 vision encoder pipeline.
pub struct VisionEncoder<B: Backend> {
    /// Patch embedding weight: reshaped from Conv3D [out, in_c, t, h, w] to 2D [hidden, patch_embed_dim]
    pub patch_embed_weight: Param<Tensor<B, 2>>,
    /// Patch embedding bias: [hidden]
    pub patch_embed_bias: Param<Tensor<B, 1>>,
    /// Learnable position embeddings: [num_position_embeddings, hidden]
    pub pos_embed: Param<Tensor<B, 2>>,
    /// Vision transformer blocks
    pub blocks: Vec<VisionBlock<B>>,
    /// Main merger (postshuffle=false)
    pub merger: PatchMerger<B>,
    /// DeepStack mergers (postshuffle=true)
    pub deepstack_mergers: Vec<PatchMerger<B>>,
    /// ViT block indices to extract intermediate features from
    pub deepstack_visual_indexes: Vec<usize>,
    /// Rotary embedding for vision attention
    pub rope: VisionRotaryEmbedding<B>,
    /// Config
    pub config: VisionConfig,
}

/// Output from the vision encoder.
pub struct VisionEncoderOutput<B: Backend> {
    /// Main merged visual features: `[num_merged_patches, out_hidden_size]`
    pub pooler_output: Tensor<B, 2>,
    /// DeepStack intermediate features: one per deepstack index, each `[num_merged_patches, out_hidden_size]`
    pub deepstack_features: Vec<Tensor<B, 2>>,
}

impl<B: Backend> VisionEncoder<B> {
    pub fn new(config: &VisionConfig, device: &Device<B>) -> Self {
        let hidden = config.hidden_size;
        let patch_embed_dim = config.patch_embed_dim();
        let eps = 1e-6;

        let blocks: Vec<VisionBlock<B>> = (0..config.depth)
            .map(|_| {
                VisionBlock::new(
                    hidden,
                    config.num_heads,
                    config.intermediate_size,
                    eps,
                    device,
                )
            })
            .collect();

        let merger = PatchMerger::new(
            hidden,
            config.out_hidden_size,
            config.spatial_merge_size,
            false,
            eps,
            device,
        );

        let deepstack_indexes = config.deepstack_visual_indexes.clone().unwrap_or_default();
        let deepstack_mergers: Vec<PatchMerger<B>> = (0..deepstack_indexes.len())
            .map(|_| {
                PatchMerger::new(
                    hidden,
                    config.out_hidden_size,
                    config.spatial_merge_size,
                    true,
                    eps,
                    device,
                )
            })
            .collect();

        let max_seq = config.num_position_embeddings;
        let head_dim = config.head_dim();
        let rope = VisionRotaryEmbedding::new(head_dim, max_seq, device);

        Self {
            patch_embed_weight: Param::from_tensor(Tensor::zeros(
                [hidden, patch_embed_dim],
                device,
            )),
            patch_embed_bias: Param::from_tensor(Tensor::zeros([hidden], device)),
            pos_embed: Param::from_tensor(Tensor::zeros(
                [config.num_position_embeddings, hidden],
                device,
            )),
            blocks,
            merger,
            deepstack_mergers,
            deepstack_visual_indexes: deepstack_indexes,
            rope,
            config: config.clone(),
        }
    }

    /// Compute 2D RoPE cos/sin for vision patches in row-major order.
    ///
    /// Matches HF: 16 inv_freq values are used for row angles and col angles,
    /// giving 32 unique angles per patch, duplicated to head_dim=64.
    fn compute_vision_rope_2d(
        &self,
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
        device: &Device<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let head_dim = self.rope.head_dim;
        let num_freqs = self.rope.inv_freq.len(); // 16
        let total_patches = grid_t * grid_h * grid_w;

        let mut cos_data = Vec::with_capacity(total_patches * head_dim);
        let mut sin_data = Vec::with_capacity(total_patches * head_dim);

        // Patches are in row-major order: (t, row, col)
        for _t in 0..grid_t {
            for row in 0..grid_h {
                for col in 0..grid_w {
                    // Row frequencies: 16 angles
                    for i in 0..num_freqs {
                        let angle = row as f64 * self.rope.inv_freq[i];
                        cos_data.push(angle.cos() as f32);
                        sin_data.push(angle.sin() as f32);
                    }
                    // Col frequencies: 16 angles
                    for i in 0..num_freqs {
                        let angle = col as f64 * self.rope.inv_freq[i];
                        cos_data.push(angle.cos() as f32);
                        sin_data.push(angle.sin() as f32);
                    }
                    // Duplicate [row_16, col_16] to [row_16, col_16, row_16, col_16] = head_dim
                    let start = cos_data.len() - 2 * num_freqs;
                    cos_data.extend_from_within(start..);
                    sin_data.extend_from_within(start..);
                }
            }
        }

        let cos =
            Tensor::<B, 1>::from_floats(&cos_data[..], device).reshape([total_patches, head_dim]);
        let sin =
            Tensor::<B, 1>::from_floats(&sin_data[..], device).reshape([total_patches, head_dim]);

        (cos, sin)
    }

    /// Bilinear interpolation of position embeddings from the stored 2D grid.
    ///
    /// The pos_embed weight is a `[num_grid^2, hidden]` table representing a 2D grid
    /// of size `num_grid x num_grid`. For patches exceeding this grid, positions are
    /// mapped via bilinear interpolation.
    fn interpolate_pos_embed(&self, grid_t: usize, grid_h: usize, grid_w: usize) -> Tensor<B, 2> {
        let [num_pos, hidden] = self.pos_embed.val().dims();
        let num_grid = (num_pos as f64).sqrt() as usize; // 48 for 2304
        let device = self.pos_embed.val().device();

        // Get pos_embed data on CPU for interpolation
        let pos_data: Vec<f32> = self.pos_embed.val().to_data().iter::<f32>().collect();

        let patches_per_frame = grid_h * grid_w;
        let total_patches = grid_t * patches_per_frame;
        let mut output = vec![0.0f32; total_patches * hidden];

        for t in 0..grid_t {
            // Map grid_h positions to [0, num_grid-1]
            // Map grid_w positions to [0, num_grid-1]
            for ph in 0..grid_h {
                let h_frac = if grid_h > 1 {
                    ph as f64 * (num_grid - 1) as f64 / (grid_h - 1) as f64
                } else {
                    0.0
                };
                let h_floor = (h_frac as usize).min(num_grid - 2);
                let h_ceil = h_floor + 1;
                let dh = h_frac - h_floor as f64;

                for pw in 0..grid_w {
                    let w_frac = if grid_w > 1 {
                        pw as f64 * (num_grid - 1) as f64 / (grid_w - 1) as f64
                    } else {
                        0.0
                    };
                    let w_floor = (w_frac as usize).min(num_grid - 2);
                    let w_ceil = w_floor + 1;
                    let dw = w_frac - w_floor as f64;

                    // 4-corner indices into pos_embed
                    let tl = h_floor * num_grid + w_floor;
                    let tr = h_floor * num_grid + w_ceil;
                    let bl = h_ceil * num_grid + w_floor;
                    let br = h_ceil * num_grid + w_ceil;

                    // Bilinear weights
                    let w_tl = ((1.0 - dh) * (1.0 - dw)) as f32;
                    let w_tr = ((1.0 - dh) * dw) as f32;
                    let w_bl = (dh * (1.0 - dw)) as f32;
                    let w_br = (dh * dw) as f32;

                    let out_idx = t * patches_per_frame + ph * grid_w + pw;
                    for d in 0..hidden {
                        output[out_idx * hidden + d] = w_tl * pos_data[tl * hidden + d]
                            + w_tr * pos_data[tr * hidden + d]
                            + w_bl * pos_data[bl * hidden + d]
                            + w_br * pos_data[br * hidden + d];
                    }
                }
            }
        }

        // Output is already in row-major (t, h, w) order, matching patch ordering
        Tensor::<B, 1>::from_floats(&output[..], &device).reshape([total_patches, hidden])
    }

    /// Forward pass: process flattened patches through vision encoder.
    ///
    /// - `patches`: `[num_patches, patch_embed_dim]` (flattened pixel patches)
    /// - `grid_t`, `grid_h`, `grid_w`: temporal/spatial grid dimensions
    ///
    /// Returns `VisionEncoderOutput` with main and deepstack features.
    pub fn forward(
        &self,
        patches: Tensor<B, 2>,
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
    ) -> VisionEncoderOutput<B> {
        let device = patches.device();
        let [_num_patches, _dim] = patches.dims();

        // 1. Linear patch embedding (Conv3D with kernel==stride acts as Linear)
        let x = patches.matmul(self.patch_embed_weight.val().transpose())
            + self.patch_embed_bias.val().unsqueeze_dim::<2>(0);

        // 2. Add positional embeddings with bilinear interpolation
        let pos_embeds = self.interpolate_pos_embed(grid_t, grid_h, grid_w);
        let x = x + pos_embeds;

        // 3. Build cu_seqlens: one sequence per temporal frame
        let patches_per_frame = grid_h * grid_w;
        let cu_seqlens: Vec<usize> = (0..=grid_t).map(|t| t * patches_per_frame).collect();

        // 4. Compute 2D RoPE cos/sin for all patches
        let cos_sin = self.compute_vision_rope_2d(grid_t, grid_h, grid_w, &device);

        // 5. Loop through blocks, collecting deepstack features
        let mut hidden_states = x;
        let mut deepstack_collected: Vec<Tensor<B, 2>> = Vec::new();

        for (i, block) in self.blocks.iter().enumerate() {
            hidden_states = block.forward(hidden_states, &cu_seqlens, Some(&cos_sin));

            if self.deepstack_visual_indexes.contains(&i) {
                deepstack_collected.push(hidden_states.clone());
            }
        }

        // 5. Main merger on final output (merge each temporal frame separately)
        let pooler_output = merge_per_frame(&hidden_states, grid_t, grid_h, grid_w, |frame| {
            self.merger.forward(frame, grid_h, grid_w)
        });

        // 6. DeepStack mergers on intermediate features
        let deepstack_features: Vec<Tensor<B, 2>> = deepstack_collected
            .into_iter()
            .zip(self.deepstack_mergers.iter())
            .map(|(feat, merger)| {
                merge_per_frame(&feat, grid_t, grid_h, grid_w, |frame| {
                    merger.forward(frame, grid_h, grid_w)
                })
            })
            .collect();

        let _ = device;
        VisionEncoderOutput {
            pooler_output,
            deepstack_features,
        }
    }

    // --- Weight loading ---

    pub fn load_patch_embed(mut self, weight: Tensor<B, 2>, bias: Tensor<B, 1>) -> Self {
        self.patch_embed_weight = Param::from_tensor(weight);
        self.patch_embed_bias = Param::from_tensor(bias);
        self
    }

    pub fn load_pos_embed(mut self, weight: Tensor<B, 2>) -> Self {
        self.pos_embed = Param::from_tensor(weight);
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_block(
        &mut self,
        block_idx: usize,
        norm1_weight: Tensor<B, 1>,
        norm1_bias: Tensor<B, 1>,
        norm2_weight: Tensor<B, 1>,
        norm2_bias: Tensor<B, 1>,
        qkv_weight: Tensor<B, 2>,
        qkv_bias: Tensor<B, 1>,
        proj_weight: Tensor<B, 2>,
        proj_bias: Tensor<B, 1>,
        fc1_weight: Tensor<B, 2>,
        fc1_bias: Tensor<B, 1>,
        fc2_weight: Tensor<B, 2>,
        fc2_bias: Tensor<B, 1>,
    ) {
        let block = &mut self.blocks[block_idx];
        block.norm1.weight = Param::from_tensor(norm1_weight);
        block.norm1.bias = Param::from_tensor(norm1_bias);
        block.norm2.weight = Param::from_tensor(norm2_weight);
        block.norm2.bias = Param::from_tensor(norm2_bias);
        block.attn.qkv.weight = Param::from_tensor(qkv_weight);
        block.attn.qkv.bias = Some(Param::from_tensor(qkv_bias));
        block.attn.proj.weight = Param::from_tensor(proj_weight);
        block.attn.proj.bias = Some(Param::from_tensor(proj_bias));
        block.mlp.fc1.weight = Param::from_tensor(fc1_weight);
        block.mlp.fc1.bias = Some(Param::from_tensor(fc1_bias));
        block.mlp.fc2.weight = Param::from_tensor(fc2_weight);
        block.mlp.fc2.bias = Some(Param::from_tensor(fc2_bias));
    }

    pub fn load_merger(
        &mut self,
        norm_weight: Tensor<B, 1>,
        norm_bias: Tensor<B, 1>,
        fc1_weight: Tensor<B, 2>,
        fc1_bias: Tensor<B, 1>,
        fc2_weight: Tensor<B, 2>,
        fc2_bias: Tensor<B, 1>,
    ) {
        self.merger.norm.weight = Param::from_tensor(norm_weight);
        self.merger.norm.bias = Param::from_tensor(norm_bias);
        self.merger.fc1.weight = Param::from_tensor(fc1_weight);
        self.merger.fc1.bias = Some(Param::from_tensor(fc1_bias));
        self.merger.fc2.weight = Param::from_tensor(fc2_weight);
        self.merger.fc2.bias = Some(Param::from_tensor(fc2_bias));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_deepstack_merger(
        &mut self,
        idx: usize,
        norm_weight: Tensor<B, 1>,
        norm_bias: Tensor<B, 1>,
        fc1_weight: Tensor<B, 2>,
        fc1_bias: Tensor<B, 1>,
        fc2_weight: Tensor<B, 2>,
        fc2_bias: Tensor<B, 1>,
    ) {
        let m = &mut self.deepstack_mergers[idx];
        m.norm.weight = Param::from_tensor(norm_weight);
        m.norm.bias = Param::from_tensor(norm_bias);
        m.fc1.weight = Param::from_tensor(fc1_weight);
        m.fc1.bias = Some(Param::from_tensor(fc1_bias));
        m.fc2.weight = Param::from_tensor(fc2_weight);
        m.fc2.bias = Some(Param::from_tensor(fc2_bias));
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

    fn to_vec_2d(t: Tensor<B, 2>) -> Vec<f32> {
        t.to_data().iter::<f32>().collect()
    }

    fn to_vec_3d(t: Tensor<B, 3>) -> Vec<f32> {
        t.to_data().iter::<f32>().collect()
    }

    // --- LayerNorm ---

    #[test]
    fn layer_norm_identity_weights() {
        let dev = device();
        let norm = LayerNorm::new(4, 1e-6, &dev);
        let x = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &dev);
        let out = norm.forward(x.clone());
        assert_eq!(out.dims(), [1, 4]);
        // Mean = 2.5, Var = 1.25, std = sqrt(1.25+eps)
        let vals = to_vec_2d(out);
        let mean = 2.5f32;
        let std = (1.25f32 + 1e-6).sqrt();
        for (i, &v) in vals.iter().enumerate() {
            let expected = ((i as f32 + 1.0) - mean) / std;
            assert!(
                (v - expected).abs() < 1e-4,
                "idx {} expected {} got {}",
                i,
                expected,
                v
            );
        }
    }

    #[test]
    fn layer_norm_with_bias() {
        let dev = device();
        let mut norm = LayerNorm::new(2, 1e-6, &dev);
        norm.weight = Param::from_tensor(Tensor::from_floats([2.0, 2.0], &dev));
        norm.bias = Param::from_tensor(Tensor::from_floats([1.0, -1.0], &dev));
        let x = Tensor::<B, 2>::from_floats([[3.0, 3.0]], &dev);
        let out = norm.forward(x);
        // Uniform input: normalized to 0, then weight*0 + bias = bias
        let vals = to_vec_2d(out);
        assert!((vals[0] - 1.0).abs() < 1e-4);
        assert!((vals[1] - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn layer_norm_3d() {
        let dev = device();
        let norm = LayerNorm::new(4, 1e-6, &dev);
        let x = Tensor::<B, 3>::ones([2, 3, 4], &dev);
        let out = norm.forward_3d(x);
        assert_eq!(out.dims(), [2, 3, 4]);
    }

    // --- VisionMLP ---

    #[test]
    fn vision_mlp_shape() {
        let dev = device();
        let mlp = VisionMLP::new(16, 32, &dev);
        let x = Tensor::<B, 2>::ones([5, 16], &dev);
        let out = mlp.forward(x);
        assert_eq!(out.dims(), [5, 16]);
    }

    // --- VisionAttention ---

    #[test]
    fn vision_attention_shape() {
        let dev = device();
        let attn = VisionAttention::new(16, 4, &dev);
        let x = Tensor::<B, 2>::ones([10, 16], &dev);
        let cu_seqlens = vec![0, 10];
        let out = attn.forward(x, &cu_seqlens, None);
        assert_eq!(out.dims(), [10, 16]);
    }

    #[test]
    fn vision_attention_multi_seq() {
        let dev = device();
        let attn = VisionAttention::new(16, 4, &dev);
        let x = Tensor::<B, 2>::ones([20, 16], &dev);
        let cu_seqlens = vec![0, 10, 20];
        let out = attn.forward(x, &cu_seqlens, None);
        assert_eq!(out.dims(), [20, 16]);
    }

    // --- VisionBlock ---

    #[test]
    fn vision_block_preserves_shape() {
        let dev = device();
        let block = VisionBlock::new(16, 4, 32, 1e-6, &dev);
        let x = Tensor::<B, 2>::ones([10, 16], &dev);
        let cu_seqlens = vec![0, 10];
        let out = block.forward(x, &cu_seqlens, None);
        assert_eq!(out.dims(), [10, 16]);
    }

    // --- PatchMerger ---

    #[test]
    fn patch_merger_pre_shuffle_shape() {
        let dev = device();
        // hidden=8, out_hidden=4, merge=2, postshuffle=false
        let merger = PatchMerger::new(8, 4, 2, false, 1e-6, &dev);
        // 4x4 grid = 16 patches -> after 2x2 merge: 2x2 = 4 merged patches
        let x = Tensor::<B, 2>::ones([16, 8], &dev);
        let out = merger.forward(x, 4, 4);
        assert_eq!(out.dims(), [4, 4]); // [4 merged patches, out_hidden=4]
    }

    #[test]
    fn patch_merger_post_shuffle_shape() {
        let dev = device();
        // hidden=8, out_hidden=4, merge=2, postshuffle=true
        let merger = PatchMerger::new(8, 4, 2, true, 1e-6, &dev);
        let x = Tensor::<B, 2>::ones([16, 8], &dev);
        let out = merger.forward(x, 4, 4);
        assert_eq!(out.dims(), [4, 4]); // [4 merged patches, out_hidden=4]
    }

    // --- spatial_merge ---

    #[test]
    fn spatial_merge_known_values() {
        let dev = device();
        // 2x2 grid, hidden=2, merge=2 -> 1 merged patch of size 8
        let x = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], &dev);
        let out = spatial_merge(x, 2, 2, 2, &dev);
        assert_eq!(out.dims(), [1, 8]);
        let vals = to_vec_2d(out);
        // Row-major: (0,0)=[1,2], (0,1)=[3,4], (1,0)=[5,6], (1,1)=[7,8]
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    // --- VisionConfig ---

    #[test]
    fn vision_config_defaults() {
        let cfg = VisionConfig::default();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.out_hidden_size, 2048);
        assert_eq!(cfg.depth, 24);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.patch_embed_dim(), 3 * 2 * 16 * 16); // 1536
        assert_eq!(cfg.merged_hidden_size(), 1024 * 4); // 4096
    }

    #[test]
    fn vision_config_deserialization() {
        let json = r#"{
            "hidden_size": 1024,
            "out_hidden_size": 2048,
            "depth": 24,
            "num_heads": 16,
            "intermediate_size": 4096,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "in_channels": 3
        }"#;
        let cfg: VisionConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.depth, 24);
    }

    // --- gelu_tanh ---

    #[test]
    fn gelu_tanh_at_zero() {
        let dev = device();
        let x = Tensor::<B, 2>::from_floats([[0.0]], &dev);
        let out = gelu_tanh(x);
        let vals = to_vec_2d(out);
        assert!((vals[0]).abs() < 1e-6, "gelu(0) should be ~0");
    }

    #[test]
    fn gelu_tanh_positive() {
        let dev = device();
        let x = Tensor::<B, 2>::from_floats([[2.0]], &dev);
        let out = gelu_tanh(x);
        let vals = to_vec_2d(out);
        // gelu(2.0) ≈ 1.9545
        assert!(
            vals[0] > 1.9 && vals[0] < 2.0,
            "gelu(2) ≈ 1.9545, got {}",
            vals[0]
        );
    }
}
