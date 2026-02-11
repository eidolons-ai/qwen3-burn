use std::collections::HashMap;
use std::ops::ControlFlow;
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

/// Reason generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    MaxTokens,
    Cancelled,
}

/// Events emitted during streaming generation.
#[derive(Debug, Clone)]
pub enum GenerationEvent {
    PrefillProgress {
        chunks_completed: usize,
        chunks_total: usize,
        prompt_tokens: usize,
    },
    Token {
        token_id: u32,
        text: String,
        tokens_generated: usize,
    },
    Done {
        tokens_generated: usize,
        total_time_secs: f64,
        prefill_time_secs: f64,
        stop_reason: StopReason,
    },
}

/// Parameters for streaming generation.
pub struct GenerationParams<'a> {
    pub prompt: &'a str,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub sampler: &'a mut Sampler,
    /// Chunk size for prefill. `None` = process entire prompt at once.
    pub prefill_chunk_size: Option<usize>,
}

/// Qwen3 language model for text generation.
///
/// Only batch size 1 is supported. KV caches and sampling are not designed for batched inference.
pub struct Qwen3<B: Backend> {
    transformer: Transformer<B>,
    rope: RotaryEmbedding<B>,
    caches: Vec<AttentionKvCache<B>>,
    config: Qwen3Config,
    max_seq_len: usize,
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
            max_seq_len,
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
    ) -> Result<GenerationOutput, String> {
        self.generate_streaming(
            tokenizer,
            GenerationParams {
                prompt,
                max_new_tokens,
                temperature,
                sampler,
                prefill_chunk_size: None,
            },
            |_| ControlFlow::Continue(()),
        )
    }

    /// Generate text with streaming callbacks and optional chunked prefill.
    ///
    /// The `callback` receives [`GenerationEvent`]s as generation progresses.
    /// Return `ControlFlow::Break(())` from the callback to cancel generation early.
    ///
    /// Returns an error if the prompt exceeds the model's `max_seq_len`.
    pub fn generate_streaming(
        &mut self,
        tokenizer: &Qwen3Tokenizer,
        params: GenerationParams,
        mut callback: impl FnMut(GenerationEvent) -> ControlFlow<()>,
    ) -> Result<GenerationOutput, String> {
        self.reset_caches();

        let tokens = tokenizer.encode(params.prompt);
        let prompt_len = tokens.len();

        if prompt_len > self.max_seq_len {
            return Err(format!(
                "prompt length ({}) exceeds max_seq_len ({})",
                prompt_len, self.max_seq_len
            ));
        }
        let mut all_tokens = tokens.clone();
        let eos_token = tokenizer.eos_token_id();

        let start_time = Instant::now();
        let mut generated_count = 0;
        let mut cancelled = false;

        // Prefill: process prompt tokens (optionally in chunks)
        let chunk_size = params.prefill_chunk_size.unwrap_or(prompt_len);
        let chunks: Vec<&[u32]> = tokens.chunks(chunk_size).collect();
        let num_chunks = chunks.len();
        let mut pos = 0;
        let mut last_logits_2d: Option<Tensor<B, 2>> = None;

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let chunk_len = chunk.len();
            let token_ids: Vec<i32> = chunk.iter().map(|&t| t as i32).collect();
            let td = TensorData::new(token_ids, [chunk_len]);
            let token_tensor = Tensor::<B, 1, Int>::from_data(td, &self.device).unsqueeze::<2>();

            let total_seq_len = pos + chunk_len;
            let mask = build_causal_mask::<B>(chunk_len, total_seq_len, &self.device);
            let logits = self.transformer.forward(
                token_tensor,
                &self.rope,
                Some(mask),
                &mut self.caches,
                pos,
            );

            // Keep logits from last chunk's final position for sampling
            let chunk_last_logits = logits
                .slice([0..1, (chunk_len - 1)..chunk_len, 0..self.config.vocab_size])
                .reshape([1, self.config.vocab_size]);
            last_logits_2d = Some(chunk_last_logits);

            pos += chunk_len;

            if callback(GenerationEvent::PrefillProgress {
                chunks_completed: chunk_idx + 1,
                chunks_total: num_chunks,
                prompt_tokens: prompt_len,
            })
            .is_break()
            {
                cancelled = true;
                break;
            }
        }

        let prefill_time = start_time.elapsed().as_secs_f64();

        if cancelled {
            let elapsed = start_time.elapsed().as_secs_f64();
            let _ = callback(GenerationEvent::Done {
                tokens_generated: 0,
                total_time_secs: elapsed,
                prefill_time_secs: prefill_time,
                stop_reason: StopReason::Cancelled,
            });
            return Ok(GenerationOutput {
                text: String::new(),
                tokens: 0,
                time: elapsed,
            });
        }

        // Early return if no tokens requested
        if params.max_new_tokens == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let _ = callback(GenerationEvent::Done {
                tokens_generated: 0,
                total_time_secs: elapsed,
                prefill_time_secs: prefill_time,
                stop_reason: StopReason::MaxTokens,
            });
            return Ok(GenerationOutput {
                text: String::new(),
                tokens: 0,
                time: elapsed,
            });
        }

        // Sample first token from prefill logits
        let logits_2d = last_logits_2d
            .ok_or_else(|| "prompt must not be empty (encoded to zero tokens)".to_string())?;
        let mut next_token = sample_token(&logits_2d, params.temperature, params.sampler);
        all_tokens.push(next_token);
        generated_count += 1;

        let stop_reason = if next_token == eos_token {
            StopReason::Eos
        } else {
            // Emit first token
            let text = tokenizer.decode(&all_tokens[prompt_len..]);
            if callback(GenerationEvent::Token {
                token_id: next_token,
                text,
                tokens_generated: generated_count,
            })
            .is_break()
            {
                let elapsed = start_time.elapsed().as_secs_f64();
                let _ = callback(GenerationEvent::Done {
                    tokens_generated: generated_count,
                    total_time_secs: elapsed,
                    prefill_time_secs: prefill_time,
                    stop_reason: StopReason::Cancelled,
                });
                return Ok(GenerationOutput {
                    text: tokenizer.decode(&all_tokens[prompt_len..]),
                    tokens: generated_count,
                    time: elapsed,
                });
            }

            // Autoregressive generation
            // No causal mask needed for single-token decode: the sole query
            // attends to all cached positions, so the mask would be all-zeros.
            let mut stop = StopReason::MaxTokens;
            for _ in 1..params.max_new_tokens {
                let _span = tracing::info_span!("decode_step", pos).entered();
                let td = TensorData::new(vec![next_token as i32], [1]);
                let token_tensor =
                    Tensor::<B, 1, Int>::from_data(td, &self.device).unsqueeze::<2>();

                let logits =
                    self.transformer
                        .forward(token_tensor, &self.rope, None, &mut self.caches, pos);

                let logits = logits.reshape([1, self.config.vocab_size]);
                next_token = sample_token(&logits, params.temperature, params.sampler);
                all_tokens.push(next_token);
                generated_count += 1;
                pos += 1;

                if next_token == eos_token {
                    stop = StopReason::Eos;
                    break;
                }

                let text = tokenizer.decode(&all_tokens[prompt_len..]);
                if callback(GenerationEvent::Token {
                    token_id: next_token,
                    text,
                    tokens_generated: generated_count,
                })
                .is_break()
                {
                    stop = StopReason::Cancelled;
                    break;
                }
            }
            stop
        };

        let elapsed = start_time.elapsed().as_secs_f64();
        let _ = callback(GenerationEvent::Done {
            tokens_generated: generated_count,
            total_time_secs: elapsed,
            prefill_time_secs: prefill_time,
            stop_reason,
        });
        Ok(GenerationOutput {
            text: tokenizer.decode(&all_tokens[prompt_len..]),
            tokens: generated_count,
            time: elapsed,
        })
    }
}

/// Apply temperature scaling and sample a token.
fn sample_token<B: Backend>(logits: &Tensor<B, 2>, temperature: f64, sampler: &mut Sampler) -> u32 {
    if temperature <= 0.0 {
        // Greedy: always pick the highest-logit token regardless of sampler
        return logits
            .clone()
            .argmax(1)
            .to_data()
            .iter::<i64>()
            .next()
            .unwrap() as u32;
    }

    let scaled = logits.clone() / temperature;
    let probs = activation::softmax(scaled, 1);

    let token_id = sampler.sample(probs);
    let token_data = token_id.to_data();
    let val = token_data.iter::<i64>().next().unwrap();
    val as u32
}

type TensorMap = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// Remove a 1D tensor from the map, consuming the f32 data (no clone).
fn take_tensor_1d<B: Backend>(
    map: &mut TensorMap,
    name: &str,
    device: &Device<B>,
) -> Result<Tensor<B, 1>, Box<dyn std::error::Error>> {
    let (data, _shape) = map
        .remove(name)
        .ok_or_else(|| format!("Tensor '{}' not found in safetensors", name))?;
    let len = data.len();
    Ok(Tensor::from_data(TensorData::new(data, [len]), device))
}

/// Remove a 2D tensor from the map (no transpose), consuming the f32 data.
fn take_tensor_2d<B: Backend>(
    map: &mut TensorMap,
    name: &str,
    device: &Device<B>,
) -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
    let (data, shape) = map
        .remove(name)
        .ok_or_else(|| format!("Tensor '{}' not found in safetensors", name))?;
    assert_eq!(
        shape.len(),
        2,
        "Expected 2D tensor for {}, got {:?}",
        name,
        shape
    );
    Ok(Tensor::from_data(
        TensorData::new(data, [shape[0], shape[1]]),
        device,
    ))
}

/// Remove a 2D Linear weight and transpose from PyTorch [out, in] to Burn [in, out].
fn take_linear_weight<B: Backend>(
    map: &mut TensorMap,
    name: &str,
    device: &Device<B>,
) -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
    Ok(take_tensor_2d(map, name, device)?.transpose())
}

/// Check whether all tensors for a given layer are present in the map.
fn layer_tensors_ready(layer_idx: usize, config: &Qwen3Config, map: &TensorMap) -> bool {
    let p = format!("model.layers.{}", layer_idx);
    let has = |suffix: &str| map.contains_key(&format!("{}.{}", p, suffix));

    // Attention + layernorm (common to all layers)
    let common_ready = has("self_attn.q_proj.weight")
        && has("self_attn.k_proj.weight")
        && has("self_attn.v_proj.weight")
        && has("self_attn.o_proj.weight")
        && has("self_attn.q_norm.weight")
        && has("self_attn.k_norm.weight")
        && has("input_layernorm.weight")
        && has("post_attention_layernorm.weight");
    if !common_ready {
        return false;
    }

    if config.is_moe_layer(layer_idx) {
        if !has("mlp.gate.weight") {
            return false;
        }
        let ne = config.num_experts.unwrap();
        for j in 0..ne {
            for proj in &["gate_proj", "up_proj", "down_proj"] {
                if !map.contains_key(&format!("{}.mlp.experts.{}.{}.weight", p, j, proj)) {
                    return false;
                }
            }
        }
    } else if !has("mlp.gate_proj.weight")
        || !has("mlp.up_proj.weight")
        || !has("mlp.down_proj.weight")
    {
        return false;
    }

    true
}

/// Load a single layer's weights from the map into the transformer, consuming (removing)
/// the f32 data from the map to free memory immediately.
fn load_layer_weights<B: Backend>(
    map: &mut TensorMap,
    transformer: Transformer<B>,
    layer_idx: usize,
    config: &Qwen3Config,
    device: &Device<B>,
) -> Result<Transformer<B>, Box<dyn std::error::Error>> {
    let prefix = format!("model.layers.{}", layer_idx);

    let q_proj_w = take_linear_weight(map, &format!("{}.self_attn.q_proj.weight", prefix), device)?;
    let k_proj_w = take_linear_weight(map, &format!("{}.self_attn.k_proj.weight", prefix), device)?;
    let v_proj_w = take_linear_weight(map, &format!("{}.self_attn.v_proj.weight", prefix), device)?;
    let o_proj_w = take_linear_weight(map, &format!("{}.self_attn.o_proj.weight", prefix), device)?;
    let q_norm_w = take_tensor_1d(map, &format!("{}.self_attn.q_norm.weight", prefix), device)?;
    let k_norm_w = take_tensor_1d(map, &format!("{}.self_attn.k_norm.weight", prefix), device)?;
    let input_ln_w = take_tensor_1d(map, &format!("{}.input_layernorm.weight", prefix), device)?;
    let post_attn_ln_w = take_tensor_1d(
        map,
        &format!("{}.post_attention_layernorm.weight", prefix),
        device,
    )?;

    if config.is_moe_layer(layer_idx) {
        let num_experts = config.num_experts.unwrap();

        // Build gate_up_proj: [num_experts, hidden, 2*moe_intermediate]
        let mut gate_up_experts: Vec<Tensor<B, 3>> = Vec::with_capacity(num_experts);
        for j in 0..num_experts {
            let gate = take_linear_weight(
                map,
                &format!("{}.mlp.experts.{}.gate_proj.weight", prefix, j),
                device,
            )?;
            let up = take_linear_weight(
                map,
                &format!("{}.mlp.experts.{}.up_proj.weight", prefix, j),
                device,
            )?;
            gate_up_experts.push(Tensor::cat(vec![gate, up], 1).unsqueeze_dim(0));
        }
        let gate_up_proj = Tensor::cat(gate_up_experts, 0);

        // Build down_proj: [num_experts, moe_intermediate, hidden]
        let mut down_experts: Vec<Tensor<B, 3>> = Vec::with_capacity(num_experts);
        for j in 0..num_experts {
            let down = take_linear_weight(
                map,
                &format!("{}.mlp.experts.{}.down_proj.weight", prefix, j),
                device,
            )?;
            down_experts.push(down.unsqueeze_dim(0));
        }
        let down_proj = Tensor::cat(down_experts, 0);

        let router_weight = take_tensor_2d(map, &format!("{}.mlp.gate.weight", prefix), device)?;

        Ok(transformer.load_moe_layer(
            layer_idx,
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
        ))
    } else {
        let gate_proj_w =
            take_linear_weight(map, &format!("{}.mlp.gate_proj.weight", prefix), device)?;
        let up_proj_w = take_linear_weight(map, &format!("{}.mlp.up_proj.weight", prefix), device)?;
        let down_proj_w =
            take_linear_weight(map, &format!("{}.mlp.down_proj.weight", prefix), device)?;

        Ok(transformer.load_layer(
            layer_idx,
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
        ))
    }
}

/// Load safetensors weights into the transformer model.
///
/// Streams weights shard-by-shard: after reading each shard file, completed layers are
/// loaded into the model immediately and their f32 data is freed. This keeps peak memory
/// close to the final model size rather than 2-3x that.
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

    let num_layers = config.num_hidden_layers;
    let tensor_needed = |name: &str| -> bool {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some(dot_pos) = rest.find('.') {
                if let Ok(layer_idx) = rest[..dot_pos].parse::<usize>() {
                    return layer_idx < num_layers;
                }
            }
        }
        true // non-layer tensors (embed_tokens, model.norm, lm_head) are always needed
    };

    let mut tensor_map: TensorMap = HashMap::new();
    let mut next_layer: usize = 0;
    let mut embed_loaded = false;
    let mut lm_head_weight: Option<Tensor<B, 2>> = None;

    for (shard_idx, path) in st_files.iter().enumerate() {
        eprintln!("Reading shard {}/{}...", shard_idx + 1, st_files.len());

        // Read shard and extract needed tensors into tensor_map.
        // The shard bytes are freed at the end of this block.
        {
            let file_bytes = std::fs::read(path)?;
            let tensors = safetensors::SafeTensors::deserialize(&file_bytes)?;
            for (name, tensor_view) in tensors.tensors() {
                if !tensor_needed(&name) {
                    continue;
                }
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
                        return Err(
                            format!("Unsupported dtype {:?} for tensor {}", dtype, name).into()
                        )
                    }
                };

                tensor_map.insert(name.to_string(), (float_data, shape));
            }
        }

        // Load embedding as soon as it's available.
        if !embed_loaded && tensor_map.contains_key("model.embed_tokens.weight") {
            eprintln!("Loading embedding weights...");
            let embed_weight =
                take_tensor_2d(&mut tensor_map, "model.embed_tokens.weight", device)?;
            if config.tie_word_embeddings {
                lm_head_weight = Some(embed_weight.clone().transpose());
            }
            transformer = transformer.load_embed_tokens(embed_weight);
            embed_loaded = true;
        }

        // Load any layers whose tensors are now all available.
        while next_layer < num_layers && layer_tensors_ready(next_layer, config, &tensor_map) {
            if next_layer.is_multiple_of(10) {
                eprintln!("Loading layer {}/{}...", next_layer, num_layers);
            }
            transformer =
                load_layer_weights(&mut tensor_map, transformer, next_layer, config, device)?;
            next_layer += 1;
        }
    }

    // Load final norm and lm_head (from whatever remains in the map).
    eprintln!("Loading final norm and lm_head...");
    let norm_weight = take_tensor_1d(&mut tensor_map, "model.norm.weight", device)?;
    transformer = transformer.load_norm(norm_weight);

    if let Some(w) = lm_head_weight {
        transformer = transformer.load_lm_head(w);
    } else {
        let w = take_linear_weight(&mut tensor_map, "lm_head.weight", device)?;
        transformer = transformer.load_lm_head(w);
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

    #[test]
    fn stop_reason_eq() {
        assert_eq!(StopReason::Eos, StopReason::Eos);
        assert_eq!(StopReason::MaxTokens, StopReason::MaxTokens);
        assert_eq!(StopReason::Cancelled, StopReason::Cancelled);
        assert_ne!(StopReason::Eos, StopReason::MaxTokens);
        assert_ne!(StopReason::Eos, StopReason::Cancelled);
        assert_ne!(StopReason::MaxTokens, StopReason::Cancelled);
    }

    #[test]
    fn generation_event_debug() {
        let event = GenerationEvent::PrefillProgress {
            chunks_completed: 1,
            chunks_total: 3,
            prompt_tokens: 100,
        };
        let s = format!("{:?}", event);
        assert!(s.contains("PrefillProgress"));
        assert!(s.contains("100"));

        let event = GenerationEvent::Token {
            token_id: 42,
            text: "hello".to_string(),
            tokens_generated: 5,
        };
        let s = format!("{:?}", event);
        assert!(s.contains("Token"));
        assert!(s.contains("hello"));

        let event = GenerationEvent::Done {
            tokens_generated: 10,
            total_time_secs: 1.5,
            prefill_time_secs: 0.3,
            stop_reason: StopReason::Eos,
        };
        let s = format!("{:?}", event);
        assert!(s.contains("Done"));
        assert!(s.contains("Eos"));
    }

    #[test]
    fn generation_params_chunk_size_none_means_no_chunking() {
        let mut sampler = Sampler::Argmax;
        let params = GenerationParams {
            prompt: "test prompt with several tokens",
            max_new_tokens: 10,
            temperature: 0.0,
            sampler: &mut sampler,
            prefill_chunk_size: None,
        };
        assert!(params.prefill_chunk_size.is_none());
    }
}
