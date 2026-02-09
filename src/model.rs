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
    build_causal_mask, AttentionKvCache, RotaryEmbedding, Transformer,
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
        let logits = self.transformer.forward(
            token_tensor,
            &self.rope,
            Some(mask),
            &mut self.caches,
            0,
        );

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
            let token_tensor =
                Tensor::<B, 1, Int>::from_data(td, &self.device).unsqueeze::<2>();

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
fn sample_token<B: Backend>(
    logits: &Tensor<B, 2>,
    temperature: f64,
    sampler: &mut Sampler,
) -> u32 {
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
        if path.extension().map_or(false, |ext| ext == "safetensors") {
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
                safetensors::Dtype::F32 => {
                    data.chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect()
                }
                safetensors::Dtype::F16 => {
                    data.chunks_exact(2)
                        .map(|chunk| {
                            half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32()
                        })
                        .collect()
                }
                safetensors::Dtype::BF16 => {
                    data.chunks_exact(2)
                        .map(|chunk| {
                            half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32()
                        })
                        .collect()
                }
                _ => return Err(format!("Unsupported dtype {:?} for tensor {}", dtype, name).into()),
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
        assert_eq!(shape.len(), 2, "Expected 2D tensor for {}, got {:?}", name, shape);
        let td = TensorData::new(data.clone(), [shape[0], shape[1]]);
        Ok(Tensor::from_data(td, device))
    };

    // Helper to get a 2D Linear weight tensor.
    // PyTorch stores Linear weights as [out, in], but Burn uses [in, out],
    // so we transpose when loading.
    let get_linear_weight = |name: &str| -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
        Ok(get_tensor_2d_raw(name)?.transpose())
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

        // MLP weights (transpose for Burn's Linear convention)
        let gate_proj_w = get_linear_weight(&format!("{}.mlp.gate_proj.weight", prefix))?;
        let up_proj_w = get_linear_weight(&format!("{}.mlp.up_proj.weight", prefix))?;
        let down_proj_w = get_linear_weight(&format!("{}.mlp.down_proj.weight", prefix))?;

        // Layer norm weights
        let input_ln_w = get_tensor_1d(&format!("{}.input_layernorm.weight", prefix))?;
        let post_attn_ln_w = get_tensor_1d(&format!("{}.post_attention_layernorm.weight", prefix))?;

        transformer = transformer.load_layer(
            i,
            q_proj_w, k_proj_w, v_proj_w, o_proj_w,
            q_norm_w, k_norm_w,
            gate_proj_w, up_proj_w, down_proj_w,
            input_ln_w, post_attn_ln_w,
        );
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
