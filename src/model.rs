use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::TensorData;
use serde::Deserialize;

use crate::sampling::Sampler;
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::{
    build_causal_mask, AttentionKvCache, MoeConfig, RotaryEmbedding, Transformer,
};

/// Qwen3 model configuration matching HuggingFace `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    // MoE fields (None for dense models)
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub decoder_sparse_step: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,
    #[serde(default)]
    pub mlp_only_layers: Option<Vec<usize>>,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_head_dim() -> usize {
    128
}

impl Qwen3Config {
    /// Load config from a `config.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// Returns true if this is a Mixture of Experts model.
    pub fn is_moe(&self) -> bool {
        self.num_experts.unwrap_or(0) > 1
    }

    /// Returns whether a given layer index uses MoE (vs dense FFN).
    /// MoE layers are determined by `decoder_sparse_step` (every N-th layer is MoE)
    /// and `mlp_only_layers` (explicit dense-only overrides).
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        if !self.is_moe() {
            return false;
        }
        // Check if this layer is in the dense-only override list
        if let Some(ref dense_layers) = self.mlp_only_layers {
            if dense_layers.contains(&layer_idx) {
                return false;
            }
        }
        // decoder_sparse_step=1 means every layer is MoE
        let step = self.decoder_sparse_step.unwrap_or(1);
        if step == 0 {
            return false;
        }
        layer_idx.is_multiple_of(step)
    }

    /// Qwen3-0.6B configuration.
    pub fn qwen3_0_6b() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            intermediate_size: 3072,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            tie_word_embeddings: true,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            decoder_sparse_step: None,
            norm_topk_prob: None,
            mlp_only_layers: None,
        }
    }

    /// Qwen3-1.7B configuration.
    pub fn qwen3_1_7b() -> Self {
        Self {
            hidden_size: 1536,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            intermediate_size: 4608,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            tie_word_embeddings: true,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            decoder_sparse_step: None,
            norm_topk_prob: None,
            mlp_only_layers: None,
        }
    }

    /// Qwen3-4B configuration.
    pub fn qwen3_4b() -> Self {
        Self {
            hidden_size: 2560,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 9728,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            tie_word_embeddings: true,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            decoder_sparse_step: None,
            norm_topk_prob: None,
            mlp_only_layers: None,
        }
    }

    /// Qwen3-8B configuration.
    pub fn qwen3_8b() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 12288,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            tie_word_embeddings: false,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            decoder_sparse_step: None,
            norm_topk_prob: None,
            mlp_only_layers: None,
        }
    }

    /// Qwen3-30B-A3B (MoE) configuration.
    pub fn qwen3_30b_a3b() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 48,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            intermediate_size: 768, // dense intermediate (not used for MoE layers)
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            tie_word_embeddings: true,
            num_experts: Some(128),
            num_experts_per_tok: Some(8),
            moe_intermediate_size: Some(768),
            decoder_sparse_step: Some(1),
            norm_topk_prob: Some(true),
            mlp_only_layers: Some(vec![]),
        }
    }

    /// Qwen3-235B-A22B (MoE) configuration.
    pub fn qwen3_235b_a22b() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 94,
            num_attention_heads: 64,
            num_key_value_heads: 4,
            intermediate_size: 1536, // dense intermediate (not used for MoE layers)
            vocab_size: 152064,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            tie_word_embeddings: false,
            num_experts: Some(128),
            num_experts_per_tok: Some(8),
            moe_intermediate_size: Some(1536),
            decoder_sparse_step: Some(1),
            norm_topk_prob: Some(true),
            mlp_only_layers: Some(vec![]),
        }
    }
}

/// Generation output.
pub struct GenerationOutput {
    pub text: String,
    pub tokens: usize,
    pub time: f64,
}

/// Qwen3 language model for text generation.
pub struct Qwen3<B: Backend> {
    transformer: Transformer<B>,
    rope: RotaryEmbedding<B>,
    caches: Vec<AttentionKvCache<B>>,
    config: Qwen3Config,
    device: Device<B>,
}

impl<B: Backend> Qwen3<B> {
    /// Load a Qwen3 model from a directory containing `config.json` and safetensors weight files.
    pub fn from_pretrained(
        model_dir: impl AsRef<Path>,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config = Qwen3Config::from_file(model_dir.join("config.json"))?;

        // Build MoE config if this is a MoE model
        let moe_config = if config.is_moe() {
            Some(MoeConfig {
                num_experts: config.num_experts.unwrap(),
                num_experts_per_tok: config.num_experts_per_tok.unwrap(),
                moe_intermediate_size: config.moe_intermediate_size.unwrap(),
                norm_topk_prob: config.norm_topk_prob.unwrap_or(true),
                mlp_only_layers: config.mlp_only_layers.clone().unwrap_or_default(),
                decoder_sparse_step: config.decoder_sparse_step.unwrap_or(1),
            })
        } else {
            None
        };

        // Create model
        let transformer = Transformer::new(
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.intermediate_size,
            config.rms_norm_eps,
            config.tie_word_embeddings,
            moe_config.as_ref(),
            device,
        );

        // Load weights from safetensors
        let transformer = load_safetensors_weights(transformer, model_dir, &config, device)?;

        let rope = RotaryEmbedding::new(config.head_dim, max_seq_len, config.rope_theta, device);

        let caches = (0..config.num_hidden_layers)
            .map(|_| {
                AttentionKvCache::new(
                    1,
                    config.num_key_value_heads,
                    max_seq_len,
                    config.head_dim,
                    device,
                )
            })
            .collect();

        Ok(Self {
            transformer,
            rope,
            caches,
            config,
            device: device.clone(),
        })
    }

    /// Reset the KV caches for a new generation.
    pub fn reset_caches(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    /// Generate text from a prompt.
    pub fn generate(
        &mut self,
        tokenizer: &Qwen3Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
        sampler: &mut Sampler,
    ) -> GenerationOutput {
        self.reset_caches();

        let tokens = tokenizer.encode(prompt);
        let mut all_tokens = tokens.clone();
        let eos_token = tokenizer.eos_token_id();

        let start_time = Instant::now();
        let mut generated_count = 0;

        // Process the full prompt
        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let td = TensorData::new(token_ids, [tokens.len()]);
        let token_tensor = Tensor::<B, 1, Int>::from_data(td, &self.device).unsqueeze::<2>();

        let seq_len = tokens.len();
        let mask = build_causal_mask::<B>(seq_len, seq_len, &self.device);
        let logits =
            self.transformer
                .forward(token_tensor, &self.rope, Some(mask), &mut self.caches, 0);

        // Sample next token from last position's logits
        let last_logits = logits
            .slice([0..1, (seq_len - 1)..seq_len, 0..self.config.vocab_size])
            .reshape([1, self.config.vocab_size]);
        let mut next_token = sample_token(&last_logits, temperature, sampler);
        all_tokens.push(next_token);
        generated_count += 1;

        if next_token == eos_token {
            let elapsed = start_time.elapsed().as_secs_f64();
            return GenerationOutput {
                text: tokenizer.decode(&all_tokens[tokens.len()..]),
                tokens: generated_count,
                time: elapsed,
            };
        }

        // Autoregressive generation
        let mut pos = seq_len;
        for _ in 1..max_new_tokens {
            let td = TensorData::new(vec![next_token as i32], [1]);
            let token_tensor = Tensor::<B, 1, Int>::from_data(td, &self.device).unsqueeze::<2>();

            let total_len = pos + 1;
            let mask = build_causal_mask::<B>(1, total_len, &self.device);
            let logits = self.transformer.forward(
                token_tensor,
                &self.rope,
                Some(mask),
                &mut self.caches,
                pos,
            );

            let logits = logits.reshape([1, self.config.vocab_size]);
            next_token = sample_token(&logits, temperature, sampler);
            all_tokens.push(next_token);
            generated_count += 1;
            pos += 1;

            if next_token == eos_token {
                break;
            }
        }

        let elapsed = start_time.elapsed().as_secs_f64();
        GenerationOutput {
            text: tokenizer.decode(&all_tokens[tokens.len()..]),
            tokens: generated_count,
            time: elapsed,
        }
    }
}

/// Apply temperature scaling and sample a token.
fn sample_token<B: Backend>(logits: &Tensor<B, 2>, temperature: f64, sampler: &mut Sampler) -> u32 {
    let logits = if temperature > 0.0 {
        let scaled = logits.clone() / temperature;
        activation::softmax(scaled, 1)
    } else {
        logits.clone()
    };

    let token_id = sampler.sample(logits);
    let token_data = token_id.to_data();
    let val = token_data.iter::<i64>().next().unwrap();
    val as u32
}

/// Load safetensors weights into the transformer model.
fn load_safetensors_weights<B: Backend>(
    mut transformer: Transformer<B>,
    model_dir: &Path,
    config: &Qwen3Config,
    device: &Device<B>,
) -> Result<Transformer<B>, Box<dyn std::error::Error>> {
    // Find all safetensors files
    let mut st_files = Vec::new();
    for entry in std::fs::read_dir(model_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            st_files.push(path);
        }
    }

    if st_files.is_empty() {
        return Err("No .safetensors files found in model directory".into());
    }

    st_files.sort();

    // Load all tensors from all files

    // We need to keep the file data alive while we use tensor views
    let mut file_data: Vec<Vec<u8>> = Vec::new();
    for path in &st_files {
        let data = std::fs::read(path)?;
        file_data.push(data);
    }

    // Load all tensor data as owned f32 vectors.
    let mut tensor_map: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    for file_bytes in &file_data {
        let tensors = safetensors::SafeTensors::deserialize(file_bytes)?;
        for (name, tensor_view) in tensors.tensors() {
            let shape: Vec<usize> = tensor_view.shape().to_vec();
            let dtype = tensor_view.dtype();
            let data = tensor_view.data();

            let float_data: Vec<f32> = match dtype {
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect(),
                safetensors::Dtype::F16 => data
                    .chunks_exact(2)
                    .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                    .collect(),
                _ => {
                    return Err(format!("Unsupported dtype {:?} for tensor {}", dtype, name).into())
                }
            };

            tensor_map.insert(name.to_string(), (float_data, shape));
        }
    }

    // Helper to get a 1D tensor from the map
    let get_tensor_1d = |name: &str| -> Result<Tensor<B, 1>, Box<dyn std::error::Error>> {
        let (data, _shape) = tensor_map
            .get(name)
            .ok_or_else(|| format!("Tensor '{}' not found in safetensors", name))?;
        let td = TensorData::new(data.clone(), [data.len()]);
        Ok(Tensor::from_data(td, device))
    };

    // Helper to get a raw 2D tensor from the map (no transpose).
    let get_tensor_2d_raw = |name: &str| -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
        let (data, shape) = tensor_map
            .get(name)
            .ok_or_else(|| format!("Tensor '{}' not found in safetensors", name))?;
        assert_eq!(
            shape.len(),
            2,
            "Expected 2D tensor for {}, got {:?}",
            name,
            shape
        );
        let td = TensorData::new(data.clone(), [shape[0], shape[1]]);
        Ok(Tensor::from_data(td, device))
    };

    // Helper to get a 2D Linear weight tensor.
    // PyTorch stores Linear weights as [out, in], but Burn uses [in, out],
    // so we transpose when loading.
    let get_linear_weight = |name: &str| -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
        Ok(get_tensor_2d_raw(name)?.transpose())
    };

    // Helper to get a raw 3D tensor (for packed expert weights).
    let get_tensor_3d_raw = |name: &str| -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
        let (data, shape) = tensor_map
            .get(name)
            .ok_or_else(|| format!("Tensor '{}' not found in safetensors", name))?;
        assert_eq!(
            shape.len(),
            3,
            "Expected 3D tensor for {}, got {:?}",
            name,
            shape
        );
        let td = TensorData::new(data.clone(), [shape[0], shape[1], shape[2]]);
        Ok(Tensor::from_data(td, device))
    };

    // Helper to get a 3D expert weight tensor and transpose the per-expert matrices.
    // PyTorch stores as [num_experts, out, in], Burn needs [num_experts, in, out].
    let get_expert_weight = |name: &str| -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
        let raw = get_tensor_3d_raw(name)?;
        // Swap dims 1 and 2 to transpose each expert's weight matrix
        Ok(raw.swap_dims(1, 2))
    };

    // Load embedding weights (no transpose needed for embeddings)
    eprintln!("Loading embedding weights...");
    let embed_weight = get_tensor_2d_raw("model.embed_tokens.weight")?;
    transformer = transformer.load_embed_tokens(embed_weight);

    // Load layer weights
    for i in 0..config.num_hidden_layers {
        if i % 10 == 0 {
            eprintln!("Loading layer {}/{}...", i, config.num_hidden_layers);
        }

        let prefix = format!("model.layers.{}", i);

        // Attention projections (transpose for Burn's Linear convention)
        let q_proj_w = get_linear_weight(&format!("{}.self_attn.q_proj.weight", prefix))?;
        let k_proj_w = get_linear_weight(&format!("{}.self_attn.k_proj.weight", prefix))?;
        let v_proj_w = get_linear_weight(&format!("{}.self_attn.v_proj.weight", prefix))?;
        let o_proj_w = get_linear_weight(&format!("{}.self_attn.o_proj.weight", prefix))?;

        // QK-Norm weights
        let q_norm_w = get_tensor_1d(&format!("{}.self_attn.q_norm.weight", prefix))?;
        let k_norm_w = get_tensor_1d(&format!("{}.self_attn.k_norm.weight", prefix))?;

        // Layer norm weights
        let input_ln_w = get_tensor_1d(&format!("{}.input_layernorm.weight", prefix))?;
        let post_attn_ln_w = get_tensor_1d(&format!("{}.post_attention_layernorm.weight", prefix))?;

        if config.is_moe_layer(i) {
            // MoE layer: load packed expert weights + router
            // gate_up_proj: [num_experts, 2*moe_intermediate, hidden] -> transposed to [num_experts, hidden, 2*moe_intermediate]
            let gate_up_proj = get_expert_weight(&format!("{}.mlp.experts.gate_up_proj", prefix))?;
            // down_proj: [num_experts, hidden, moe_intermediate] -> transposed to [num_experts, moe_intermediate, hidden]
            let down_proj = get_expert_weight(&format!("{}.mlp.experts.down_proj", prefix))?;
            // router.weight: [num_experts, hidden] (no transpose needed, used as-is for matmul)
            let router_weight = get_tensor_2d_raw(&format!("{}.mlp.router.weight", prefix))?;

            transformer = transformer.load_moe_layer(
                i,
                q_proj_w,
                k_proj_w,
                v_proj_w,
                o_proj_w,
                q_norm_w,
                k_norm_w,
                gate_up_proj,
                down_proj,
                router_weight,
                input_ln_w,
                post_attn_ln_w,
            );
        } else {
            // Dense layer: load individual MLP weights
            let gate_proj_w = get_linear_weight(&format!("{}.mlp.gate_proj.weight", prefix))?;
            let up_proj_w = get_linear_weight(&format!("{}.mlp.up_proj.weight", prefix))?;
            let down_proj_w = get_linear_weight(&format!("{}.mlp.down_proj.weight", prefix))?;

            transformer = transformer.load_layer(
                i,
                q_proj_w,
                k_proj_w,
                v_proj_w,
                o_proj_w,
                q_norm_w,
                k_norm_w,
                gate_proj_w,
                up_proj_w,
                down_proj_w,
                input_ln_w,
                post_attn_ln_w,
            );
        }
    }

    // Load final norm
    eprintln!("Loading final norm and lm_head...");
    let norm_weight = get_tensor_1d("model.norm.weight")?;
    transformer = transformer.load_norm(norm_weight);

    // Load lm_head (may be tied with embeddings)
    // For lm_head, the weight maps input [d_model] -> [vocab_size],
    // so it's a Linear layer needing transpose.
    // When tied with embeddings, we use the embedding weight which is [vocab_size, d_model],
    // which happens to already be the correct transposed form for Burn's Linear [in, out] = [d_model, vocab_size].
    if config.tie_word_embeddings {
        let embed_weight = get_tensor_2d_raw("model.embed_tokens.weight")?;
        // Embedding is [vocab_size, d_model], need [d_model, vocab_size] for Linear
        transformer = transformer.load_lm_head(embed_weight.transpose());
    } else {
        let lm_head_weight = get_linear_weight("lm_head.weight")?;
        transformer = transformer.load_lm_head(lm_head_weight);
    }

    eprintln!("Model loaded successfully.");
    Ok(transformer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_config_is_not_moe() {
        let config = Qwen3Config::qwen3_0_6b();
        assert!(!config.is_moe());
        assert!(!config.is_moe_layer(0));
        assert!(!config.is_moe_layer(10));
    }

    #[test]
    fn moe_config_is_moe() {
        let config = Qwen3Config::qwen3_30b_a3b();
        assert!(config.is_moe());
    }

    #[test]
    fn moe_every_layer_when_step_is_1() {
        let config = Qwen3Config::qwen3_30b_a3b();
        assert_eq!(config.decoder_sparse_step, Some(1));
        for i in 0..config.num_hidden_layers {
            assert!(config.is_moe_layer(i), "layer {} should be MoE", i);
        }
    }

    #[test]
    fn moe_layer_respects_mlp_only_layers() {
        let mut config = Qwen3Config::qwen3_30b_a3b();
        config.mlp_only_layers = Some(vec![0, 5, 10]);
        assert!(!config.is_moe_layer(0));
        assert!(config.is_moe_layer(1));
        assert!(!config.is_moe_layer(5));
        assert!(!config.is_moe_layer(10));
        assert!(config.is_moe_layer(11));
    }

    #[test]
    fn moe_layer_with_sparse_step_2() {
        let mut config = Qwen3Config::qwen3_30b_a3b();
        config.decoder_sparse_step = Some(2);
        // step=2: layers 0, 2, 4 are MoE; layers 1, 3, 5 are dense
        assert!(config.is_moe_layer(0));
        assert!(!config.is_moe_layer(1));
        assert!(config.is_moe_layer(2));
        assert!(!config.is_moe_layer(3));
    }

    #[test]
    fn moe_layer_step_0_means_no_moe() {
        let mut config = Qwen3Config::qwen3_30b_a3b();
        config.decoder_sparse_step = Some(0);
        for i in 0..config.num_hidden_layers {
            assert!(!config.is_moe_layer(i));
        }
    }

    #[test]
    fn serde_dense_config() {
        let json = r#"{
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 3072,
            "vocab_size": 151936,
            "max_position_embeddings": 40960
        }"#;
        let config: Qwen3Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 28);
        assert!(!config.is_moe());
        // Defaults
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert_eq!(config.head_dim, 128);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.num_experts, None);
    }

    #[test]
    fn serde_moe_config() {
        let json = r#"{
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 768,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "decoder_sparse_step": 1,
            "norm_topk_prob": true,
            "mlp_only_layers": []
        }"#;
        let config: Qwen3Config = serde_json::from_str(json).unwrap();
        assert!(config.is_moe());
        assert_eq!(config.num_experts, Some(128));
        assert_eq!(config.num_experts_per_tok, Some(8));
        assert_eq!(config.moe_intermediate_size, Some(768));
        assert_eq!(config.norm_topk_prob, Some(true));
        assert_eq!(config.mlp_only_layers, Some(vec![]));
    }

    #[test]
    fn preset_dense_configs() {
        let c = Qwen3Config::qwen3_0_6b();
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.num_hidden_layers, 28);
        assert!(c.tie_word_embeddings);
        assert!(!c.is_moe());

        let c = Qwen3Config::qwen3_8b();
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_hidden_layers, 36);
        assert!(!c.tie_word_embeddings);
        assert!(!c.is_moe());
    }

    #[test]
    fn preset_moe_configs() {
        let c = Qwen3Config::qwen3_30b_a3b();
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.num_hidden_layers, 48);
        assert_eq!(c.num_experts, Some(128));
        assert_eq!(c.num_experts_per_tok, Some(8));
        assert!(c.is_moe());

        let c = Qwen3Config::qwen3_235b_a22b();
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_hidden_layers, 94);
        assert_eq!(c.num_experts, Some(128));
        assert!(c.is_moe());
    }
}
