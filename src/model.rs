use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::Path;
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::TensorData;
use serde::Deserialize;

use crate::gguf;
use crate::sampling::Sampler;
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::{
    build_causal_mask, AttentionKvCache, MoeConfig, RotaryEmbedding, Transformer,
};

/// Weight quantization mode applied after loading.
///
/// Uses Burn's native `Quantizer` to store weights in packed quantized format
/// (`PackedU32`) for real memory savings. Requires a GPU backend (WGPU, CUDA);
/// the NdArray/CPU backend does not support `PackedU32` storage. A selective
/// wrapper skips 1D tensors (RMSNorm weights) and embedding weights.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum QuantizationMode {
    /// Auto-detect from GGUF source (Q8_0→Int8, Q4_0→Int4); no quantization for SafeTensors.
    #[default]
    Auto,
    /// No quantization — always load as f32. Works on all backends including NdArray/CPU.
    None,
    /// INT8 symmetric per-block quantization (Q8S, block size 32). Requires GPU backend.
    Int8,
    /// INT4 symmetric per-block quantization (Q4S, block size 32). Requires GPU backend.
    /// NOTE: Currently broken on WGPU/Metal — Q4S matmul panics when M > 1 (i.e. during
    /// prefill). See https://github.com/tracel-ai/burn/issues/4492
    Int4,
}

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
        quantization: QuantizationMode,
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

        // Quantize weights if requested (Auto means no quantization for SafeTensors)
        let effective = match quantization {
            QuantizationMode::Auto => QuantizationMode::None,
            other => other,
        };
        let transformer = apply_quantization(transformer, effective);

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

    /// Load a Qwen3 model from a GGUF file.
    ///
    /// Only requires a single `.gguf` file; config is extracted from GGUF metadata.
    /// Supported GGUF quantization types: F32, F16, BF16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.
    pub fn from_gguf(
        gguf_path: impl AsRef<Path>,
        max_seq_len: usize,
        quantization: QuantizationMode,
        device: &Device<B>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (gguf_file, mut file) = gguf::GgufFile::open(gguf_path)?;

        // Extract config from GGUF metadata
        let config = gguf::extract_config(&gguf_file)?;
        eprintln!(
            "GGUF config: hidden={}, layers={}, heads={}, kv_heads={}, vocab={}{}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.vocab_size,
            if config.is_moe() {
                format!(", MoE experts={}", config.num_experts.unwrap_or(0))
            } else {
                String::new()
            }
        );

        // Build MoE config if needed
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

        // Create model skeleton
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

        // Resolve quantization mode: auto-detect from GGUF when Auto
        let detected = detect_gguf_quantization(&gguf_file);
        let resolved_quant = match quantization {
            QuantizationMode::Auto => detected,
            other => other,
        };

        // Build QuantScheme for per-tensor quantized loading
        let quant_scheme = {
            use burn::tensor::quantization::{QuantLevel, QuantScheme, QuantValue};
            match resolved_quant {
                QuantizationMode::Auto | QuantizationMode::None => None,
                QuantizationMode::Int8 => {
                    eprintln!("Loading GGUF with per-tensor INT8 quantization...");
                    Some(
                        QuantScheme::default()
                            .with_value(QuantValue::Q8S)
                            .with_level(QuantLevel::block([32])),
                    )
                }
                QuantizationMode::Int4 => {
                    eprintln!("Loading GGUF with per-tensor INT4 quantization...");
                    Some(
                        QuantScheme::default()
                            .with_value(QuantValue::Q4S)
                            .with_level(QuantLevel::block([32])),
                    )
                }
            }
        };

        // Load weights from GGUF (per-tensor quantization applied during loading if scheme is set)
        let per_tensor_quantized = quant_scheme.is_some();
        let transformer = load_gguf_weights(
            transformer,
            &gguf_file,
            &mut file,
            &config,
            quant_scheme,
            device,
        )?;

        // Only apply whole-model quantization if per-tensor wasn't already done
        let transformer = if per_tensor_quantized {
            transformer
        } else {
            apply_quantization(transformer, resolved_quant)
        };

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

        // Accumulate decoded text incrementally to avoid re-decoding the entire
        // generated sequence on every step (which is O(n²) in total).
        let mut decoded_text = tokenizer.decode(&[next_token]);

        let stop_reason = if next_token == eos_token {
            StopReason::Eos
        } else {
            // Emit first token
            if callback(GenerationEvent::Token {
                token_id: next_token,
                text: decoded_text.clone(),
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
                    text: decoded_text,
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

                decoded_text.push_str(&tokenizer.decode(&[next_token]));
                if callback(GenerationEvent::Token {
                    token_id: next_token,
                    text: decoded_text.clone(),
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
            text: decoded_text,
            tokens: generated_count,
            time: elapsed,
        })
    }
}

/// Apply temperature scaling and sample a token.
///
/// Logits are extracted to f64 on CPU before softmax so that f16 backends
/// don't overflow/underflow over large vocabularies.
fn sample_token<B: Backend>(logits: &Tensor<B, 2>, temperature: f64, sampler: &mut Sampler) -> u32 {
    let logits_f64: Vec<f64> = logits.to_data().iter::<f64>().collect();

    if temperature <= 0.0 {
        return logits_f64
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as u32;
    }

    // Temperature scaling + softmax in f64
    let max = logits_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits_f64
        .iter()
        .map(|&x| ((x - max) / temperature).exp())
        .collect();
    let sum: f64 = exp.iter().sum();
    let probs: Vec<f64> = exp.iter().map(|&x| x / sum).collect();

    sampler.sample_probs(&probs)
}

type TensorMap = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// Convert a single FP8 E4M3FN byte to f32.
/// Format: 1 sign, 4 exponent (bias=7), 3 mantissa. No infinities; 0x7F/0xFF are NaN.
#[inline]
pub(crate) fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = (b >> 3) & 0xF;
    let mantissa = b & 0x7;

    if exp == 0xF && mantissa == 0x7 {
        return f32::NAN;
    }

    if exp == 0 {
        // Subnormal: value = (-1)^sign * 2^(1-bias) * (mantissa / 8)
        if mantissa == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Direct computation for subnormals (rare, only 7 values per sign)
        let val = (mantissa as f32) * (1.0 / 8.0) * (1.0 / 64.0); // 2^(1-7) = 2^-6 = 1/64
        return if sign == 1 { -val } else { val };
    }

    // Normal: value = (-1)^sign * 2^(exp-7) * (1 + mantissa/8)
    // Rebias: f32_bias(127) - fp8_bias(7) = 120
    let fexp = (exp as u32) + 120;
    let bits = (sign as u32) << 31 | fexp << 23 | (mantissa as u32) << 20;
    f32::from_bits(bits)
}

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
        let gate_up_proj_w = Tensor::cat(vec![gate_proj_w, up_proj_w], 1);
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
            gate_up_proj_w,
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
                    safetensors::Dtype::F8_E4M3 => {
                        data.iter().map(|&b| fp8_e4m3_to_f32(b)).collect()
                    }
                    _ => {
                        return Err(
                            format!("Unsupported dtype {:?} for tensor {}", dtype, name).into()
                        )
                    }
                };

                tensor_map.insert(name.to_string(), (float_data, shape));
            }
        }

        // Apply FP8 block-wise dequantization: for each weight_scale_inv tensor,
        // multiply the corresponding weight by the per-block scale factors.
        let scale_keys: Vec<String> = tensor_map
            .keys()
            .filter(|k| k.ends_with(".weight_scale_inv"))
            .cloned()
            .collect();
        for scale_key in scale_keys {
            let weight_key = scale_key.replace(".weight_scale_inv", ".weight");
            let (scale_data, scale_shape) = match tensor_map.remove(&scale_key) {
                Some(v) => v,
                None => continue,
            };
            if let Some((weight_data, weight_shape)) = tensor_map.get_mut(&weight_key) {
                debug_assert_eq!(scale_shape.len(), 2);
                let block_rows = scale_shape[0];
                let block_cols = scale_shape[1];
                let rows = weight_shape[0];
                let cols = weight_shape[1];
                let brow_size = rows / block_rows;
                let bcol_size = cols / block_cols;
                for br in 0..block_rows {
                    for bc in 0..block_cols {
                        let scale = scale_data[br * block_cols + bc];
                        let row_start = br * brow_size;
                        let col_start = bc * bcol_size;
                        for r in row_start..row_start + brow_size {
                            let offset = r * cols + col_start;
                            for v in &mut weight_data[offset..offset + bcol_size] {
                                *v *= scale;
                            }
                        }
                    }
                }
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

/// Quantize f32 data to INT8 symmetric (Q8S) on CPU.
///
/// Returns `(int8_values, scales)` where scales has one entry per block.
/// Each block of `block_size` elements gets `scale = max_abs / 127.0`.
fn quantize_q8s(data: &[f32], block_size: usize) -> (Vec<i8>, Vec<f32>) {
    let num_blocks = data.len() / block_size;
    let mut values = Vec::with_capacity(data.len());
    let mut scales = Vec::with_capacity(num_blocks);

    for block in data.chunks_exact(block_size) {
        let max_abs = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales.push(scale);
        let inv_scale = 1.0 / scale;
        for &v in block {
            let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            values.push(q);
        }
    }

    (values, scales)
}

/// Quantize f32 data to INT4 symmetric (Q4S) on CPU.
///
/// Returns `(int8_values, scales)` where int8 values are in [-7, 7] range.
/// Burn's `TensorData::quantized` handles packing i8 into PackedU32 for Q4S.
fn quantize_q4s(data: &[f32], block_size: usize) -> (Vec<i8>, Vec<f32>) {
    let num_blocks = data.len() / block_size;
    let mut values = Vec::with_capacity(data.len());
    let mut scales = Vec::with_capacity(num_blocks);

    for block in data.chunks_exact(block_size) {
        let max_abs = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
        scales.push(scale);
        let inv_scale = 1.0 / scale;
        for &v in block {
            let q = (v * inv_scale).round().clamp(-7.0, 7.0) as i8;
            values.push(q);
        }
    }

    (values, scales)
}

/// Create a 2D Burn tensor from f32 data, transposing on CPU and quantizing directly.
///
/// Input data is in `[out_features, in_features]` (row-major), and the resulting
/// tensor is `[in_features, out_features]` with quantized storage.
fn make_2d_quantized<B: Backend>(
    data: Vec<f32>,
    shape: &[usize],
    scheme: &burn::tensor::quantization::QuantScheme,
    device: &Device<B>,
) -> Tensor<B, 2> {
    use burn::tensor::quantization::{QuantLevel, QuantValue};

    let (rows, cols) = (shape[0], shape[1]);

    // Transpose on CPU: [rows, cols] -> [cols, rows]
    let mut transposed = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }

    let block_size = match &scheme.level {
        QuantLevel::Block(bs) => bs.as_slice()[0] as usize,
        QuantLevel::Tensor => transposed.len(),
    };

    let (int8_vals, scales) = match scheme.value {
        QuantValue::Q8S | QuantValue::Q8F => quantize_q8s(&transposed, block_size),
        QuantValue::Q4S | QuantValue::Q4F => quantize_q4s(&transposed, block_size),
        _ => quantize_q8s(&transposed, block_size), // fallback
    };

    let td = TensorData::quantized(int8_vals, [cols, rows], *scheme, &scales);
    Tensor::from_data(td, device)
}

/// Detect quantization mode from GGUF tensor dtypes.
///
/// Checks a representative weight tensor to determine the source quantization.
pub(crate) fn detect_gguf_quantization(gguf_file: &gguf::GgufFile) -> QuantizationMode {
    // Check the first attention weight tensor
    let dtype = gguf_file
        .tensor_dtype("blk.0.attn_q.weight")
        .or_else(|| gguf_file.tensor_dtype("blk.0.attn_k.weight"));

    match dtype {
        Some(gguf::GgufDtype::Q8_0) => QuantizationMode::Int8,
        Some(gguf::GgufDtype::Q4_0) => QuantizationMode::Int4,
        // K-quant types: higher precision sources map to Int8, lower to Int4
        Some(gguf::GgufDtype::Q5_K) | Some(gguf::GgufDtype::Q6_K) => QuantizationMode::Int8,
        Some(gguf::GgufDtype::Q2_K) | Some(gguf::GgufDtype::Q3_K) | Some(gguf::GgufDtype::Q4_K) => {
            QuantizationMode::Int4
        }
        _ => QuantizationMode::None,
    }
}

/// Load GGUF weights into the transformer model.
///
/// When `quant_scheme` is `Some`, 2D linear weights are quantized per-tensor on CPU
/// before being sent to the device, avoiding a full f32 model on GPU.
pub(crate) fn load_gguf_weights<B: Backend>(
    mut transformer: Transformer<B>,
    gguf_file: &gguf::GgufFile,
    file: &mut std::fs::File,
    config: &Qwen3Config,
    quant_scheme: Option<burn::tensor::quantization::QuantScheme>,
    device: &Device<B>,
) -> Result<Transformer<B>, Box<dyn std::error::Error>> {
    // Helper: read a tensor from GGUF and return as f32 vec with reversed dims (PyTorch convention)
    let read_tensor = |file: &mut std::fs::File,
                       name: &str|
     -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let info = gguf_file
            .tensors
            .get(name)
            .ok_or_else(|| format!("GGUF tensor '{}' not found", name))?;
        let data = gguf_file.read_tensor_data(file, info)?;
        // Reverse dims from GGUF (innermost-first) to PyTorch (outermost-first)
        let shape: Vec<usize> = info.dims.iter().rev().copied().collect();
        Ok((data, shape))
    };

    // Helper: create a 1D Burn tensor
    let make_1d = |data: Vec<f32>, device: &Device<B>| -> Tensor<B, 1> {
        let len = data.len();
        Tensor::from_data(TensorData::new(data, [len]), device)
    };

    // Helper: create a 2D Burn tensor
    let make_2d = |data: Vec<f32>, shape: &[usize], device: &Device<B>| -> Tensor<B, 2> {
        Tensor::from_data(TensorData::new(data, [shape[0], shape[1]]), device)
    };

    // Helper: create a 2D linear weight tensor (transposed), quantized if scheme is set.
    // Input shape is [out, in] (PyTorch convention); result is [in, out] (Burn convention).
    let make_2d_linear = |data: Vec<f32>, shape: &[usize], device: &Device<B>| -> Tensor<B, 2> {
        if let Some(ref scheme) = quant_scheme {
            make_2d_quantized(data, shape, scheme, device)
        } else {
            let t = Tensor::from_data(TensorData::new(data, [shape[0], shape[1]]), device);
            t.transpose()
        }
    };

    // Helper: create a 3D Burn tensor
    let make_3d = |data: Vec<f32>, shape: &[usize], device: &Device<B>| -> Tensor<B, 3> {
        Tensor::from_data(
            TensorData::new(data, [shape[0], shape[1], shape[2]]),
            device,
        )
    };

    // Load embedding
    eprintln!("Loading embedding weights...");
    let (embed_data, embed_shape) = read_tensor(file, "token_embd.weight")?;
    let embed_weight = make_2d(embed_data, &embed_shape, device);
    transformer = transformer.load_embed_tokens(embed_weight);

    // Load layers
    for i in 0..config.num_hidden_layers {
        if i % 10 == 0 {
            eprintln!("Loading layer {}/{}...", i, config.num_hidden_layers);
        }

        // Attention projections — GGUF stores [out, in] (PyTorch convention after dim reversal),
        // Burn needs [in, out], so transpose. When quantizing, this is done on CPU.
        let (q_data, q_shape) = read_tensor(file, &format!("blk.{}.attn_q.weight", i))?;
        let q_proj_w = make_2d_linear(q_data, &q_shape, device);

        let (k_data, k_shape) = read_tensor(file, &format!("blk.{}.attn_k.weight", i))?;
        let k_proj_w = make_2d_linear(k_data, &k_shape, device);

        let (v_data, v_shape) = read_tensor(file, &format!("blk.{}.attn_v.weight", i))?;
        let v_proj_w = make_2d_linear(v_data, &v_shape, device);

        let (o_data, o_shape) = read_tensor(file, &format!("blk.{}.attn_output.weight", i))?;
        let o_proj_w = make_2d_linear(o_data, &o_shape, device);

        // QK-Norm weights (1D)
        let (qn_data, _) = read_tensor(file, &format!("blk.{}.attn_q_norm.weight", i))?;
        let q_norm_w = make_1d(qn_data, device);

        let (kn_data, _) = read_tensor(file, &format!("blk.{}.attn_k_norm.weight", i))?;
        let k_norm_w = make_1d(kn_data, device);

        // Layer norms (1D)
        let (iln_data, _) = read_tensor(file, &format!("blk.{}.attn_norm.weight", i))?;
        let input_ln_w = make_1d(iln_data, device);

        let (pln_data, _) = read_tensor(file, &format!("blk.{}.ffn_norm.weight", i))?;
        let post_attn_ln_w = make_1d(pln_data, device);

        if config.is_moe_layer(i) {
            // MoE layer: GGUF stores gate and up as separate packed 3D tensors.
            // After dim reversal, shapes are [num_experts, moe_intermediate, hidden].
            // We need to transpose each expert [moe_intermediate, hidden] -> [hidden, moe_intermediate],
            // then concatenate gate and up along dim 2 to get [num_experts, hidden, 2*moe_intermediate].
            let (gate_data, gate_shape) =
                read_tensor(file, &format!("blk.{}.ffn_gate_exps.weight", i))?;
            let gate_3d = make_3d(gate_data, &gate_shape, device).swap_dims(1, 2);

            let (up_data, up_shape) = read_tensor(file, &format!("blk.{}.ffn_up_exps.weight", i))?;
            let up_3d = make_3d(up_data, &up_shape, device).swap_dims(1, 2);

            // Concatenate gate and up along dim 2: [num_experts, hidden, 2*moe_intermediate]
            let gate_up_proj = burn::tensor::Tensor::cat(vec![gate_3d, up_3d], 2);

            // down_proj: [num_experts, hidden, moe_intermediate] -> swap to [num_experts, moe_intermediate, hidden]
            let (down_data, down_shape) =
                read_tensor(file, &format!("blk.{}.ffn_down_exps.weight", i))?;
            let down_proj = make_3d(down_data, &down_shape, device).swap_dims(1, 2);

            // Router weight: [num_experts, hidden] (no transpose needed)
            let (router_data, router_shape) =
                read_tensor(file, &format!("blk.{}.ffn_gate_inp.weight", i))?;
            let router_weight = make_2d(router_data, &router_shape, device);

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
            // Dense layer: fuse gate+up at f32 level before transpose/quantization
            let (mut gate_up_data, gate_shape) =
                read_tensor(file, &format!("blk.{}.ffn_gate.weight", i))?;
            let (up_data, up_shape) = read_tensor(file, &format!("blk.{}.ffn_up.weight", i))?;
            // Both are [intermediate, hidden]; concatenate rows → [2*intermediate, hidden]
            gate_up_data.extend_from_slice(&up_data);
            let gate_up_shape = vec![gate_shape[0] + up_shape[0], gate_shape[1]];
            let gate_up_proj_w = make_2d_linear(gate_up_data, &gate_up_shape, device);

            let (down_data, down_shape) = read_tensor(file, &format!("blk.{}.ffn_down.weight", i))?;
            let down_proj_w = make_2d_linear(down_data, &down_shape, device);

            transformer = transformer.load_layer(
                i,
                q_proj_w,
                k_proj_w,
                v_proj_w,
                o_proj_w,
                q_norm_w,
                k_norm_w,
                gate_up_proj_w,
                down_proj_w,
                input_ln_w,
                post_attn_ln_w,
            );
        }
    }

    // Load final norm
    eprintln!("Loading final norm and lm_head...");
    let (norm_data, _) = read_tensor(file, "output_norm.weight")?;
    let norm_weight = make_1d(norm_data, device);
    transformer = transformer.load_norm(norm_weight);

    // Load lm_head
    if config.tie_word_embeddings {
        // Re-read embedding and transpose for lm_head: [vocab, hidden] -> [hidden, vocab]
        let (embed_data, embed_shape) = read_tensor(file, "token_embd.weight")?;
        let embed_weight = make_2d_linear(embed_data, &embed_shape, device);
        transformer = transformer.load_lm_head(embed_weight);
    } else {
        let (lm_data, lm_shape) = read_tensor(file, "output.weight")?;
        let lm_head_weight = make_2d_linear(lm_data, &lm_shape, device);
        transformer = transformer.load_lm_head(lm_head_weight);
    }

    eprintln!("Model loaded successfully.");
    Ok(transformer)
}

/// A selective quantizer wrapping Burn's native `Quantizer`.
///
/// Skips parameters that are sensitive to quantization:
/// - 1D tensors (RMSNorm weights)
/// - Embedding weights (`embed_tokens`)
///
/// Also forces tensors to be contiguous before quantization. Model weights are
/// loaded as `[out, in]` then `.transpose()`d to `[in, out]`, which creates
/// non-contiguous memory views. Burn's block quantization computes scales over
/// physical memory blocks, but dequantization applies them over logical blocks,
/// causing catastrophic errors (~500x) for non-contiguous tensors.
struct SelectiveQuantizer {
    quantizer: burn::module::Quantizer,
    /// Module path segments tracked via enter/exit.
    path: Vec<String>,
}

impl<B: Backend> burn::module::ModuleMapper<B> for SelectiveQuantizer {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(
        &mut self,
        param: burn::module::Param<Tensor<B, D>>,
    ) -> burn::module::Param<Tensor<B, D>> {
        // Skip 1D tensors (norm weights — tiny and critical for stability)
        if D < 2 {
            return param;
        }

        // Skip embedding weights (quantizing the lookup table degrades token representations)
        if self.path.contains(&"embed_tokens".to_string()) {
            return param;
        }

        // TODO: Remove once https://github.com/tracel-ai/burn/issues/4491 is fixed.
        // Force contiguous layout before quantization. Transposed weights have
        // non-contiguous memory which causes Burn's quantization to produce
        // catastrophically wrong scales. Round-tripping through TensorData
        // forces a contiguous copy.
        let tensor = param.val();
        let data = tensor.to_data();
        let contiguous = Tensor::from_data(data, &tensor.device());
        let param = burn::module::Param::from_tensor(contiguous);

        self.quantizer.map_float(param)
    }
}

/// Apply quantization to the transformer model using Burn's native quantization.
///
/// Weights are stored in quantized format with `PackedU32` storage for real
/// memory savings on GPU backends (WGPU, CUDA). NdArray does not support
/// `PackedU32`; quantization requires a GPU backend.
pub(crate) fn apply_quantization<B: Backend>(
    transformer: Transformer<B>,
    mode: QuantizationMode,
) -> Transformer<B> {
    use burn::module::Module;
    use burn::tensor::quantization::{Calibration, QuantLevel, QuantScheme, QuantValue};

    let scheme = match mode {
        QuantizationMode::Auto | QuantizationMode::None => return transformer,
        QuantizationMode::Int8 => {
            eprintln!("Quantizing weights to INT8...");
            QuantScheme::default()
                .with_value(QuantValue::Q8S)
                .with_level(QuantLevel::block([32]))
        }
        QuantizationMode::Int4 => {
            eprintln!("Quantizing weights to INT4...");
            QuantScheme::default()
                .with_value(QuantValue::Q4S)
                .with_level(QuantLevel::block([32]))
        }
    };

    let mut quantizer = SelectiveQuantizer {
        quantizer: burn::module::Quantizer {
            calibration: Calibration::MinMax,
            scheme,
        },
        path: Vec::new(),
    };
    transformer.map(&mut quantizer)
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

    #[test]
    fn quantization_mode_default_is_auto() {
        assert_eq!(QuantizationMode::default(), QuantizationMode::Auto);
    }

    #[test]
    fn quantization_none_is_identity() {
        use crate::transformer::{
            build_causal_mask, AttentionKvCache, RotaryEmbedding, Transformer,
        };
        use burn::backend::NdArray;

        type B = NdArray;
        let dev: <B as burn::prelude::Backend>::Device = Default::default();

        let transformer = Transformer::<B>::new(100, 32, 2, 4, 2, 8, 64, 1e-6, false, None, &dev);
        let transformer = apply_quantization(transformer, QuantizationMode::None);

        let rope = RotaryEmbedding::new(8, 16, 10000.0, &dev);
        let mut caches: Vec<AttentionKvCache<B>> = (0..2)
            .map(|_| AttentionKvCache::new(1, 2, 16, 8, &dev))
            .collect();
        let tokens = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::from_data(
            TensorData::new(vec![1i32, 2, 3], [1, 3]),
            &dev,
        );
        let mask = build_causal_mask::<B>(3, 3, &dev);
        let out = transformer.forward(tokens, &rope, Some(mask), &mut caches, 0);
        assert_eq!(out.dims(), [1, 3, 100]);
    }

    #[test]
    #[ignore] // PackedU32 storage requires a GPU backend (WGPU/CUDA)
    fn quantization_int8_preserves_shape() {
        use crate::transformer::{
            build_causal_mask, AttentionKvCache, RotaryEmbedding, Transformer,
        };
        use burn::backend::NdArray;

        type B = NdArray;
        let dev: <B as burn::prelude::Backend>::Device = Default::default();

        let transformer = Transformer::<B>::new(100, 32, 2, 4, 2, 8, 64, 1e-6, false, None, &dev);
        let transformer = apply_quantization(transformer, QuantizationMode::Int8);

        let rope = RotaryEmbedding::new(8, 16, 10000.0, &dev);
        let mut caches: Vec<AttentionKvCache<B>> = (0..2)
            .map(|_| AttentionKvCache::new(1, 2, 16, 8, &dev))
            .collect();
        let tokens = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::from_data(
            TensorData::new(vec![1i32, 2, 3], [1, 3]),
            &dev,
        );
        let mask = build_causal_mask::<B>(3, 3, &dev);
        let out = transformer.forward(tokens, &rope, Some(mask), &mut caches, 0);
        assert_eq!(out.dims(), [1, 3, 100]);
    }

    #[test]
    #[ignore] // INT4 uses PackedU32 storage which requires a GPU backend (WGPU/CUDA)
    fn quantization_int4_preserves_shape() {
        use crate::transformer::{
            build_causal_mask, AttentionKvCache, RotaryEmbedding, Transformer,
        };
        use burn::backend::NdArray;

        type B = NdArray;
        let dev: <B as burn::prelude::Backend>::Device = Default::default();

        let transformer = Transformer::<B>::new(100, 32, 2, 4, 2, 8, 64, 1e-6, false, None, &dev);
        let transformer = apply_quantization(transformer, QuantizationMode::Int4);

        let rope = RotaryEmbedding::new(8, 16, 10000.0, &dev);
        let mut caches: Vec<AttentionKvCache<B>> = (0..2)
            .map(|_| AttentionKvCache::new(1, 2, 16, 8, &dev))
            .collect();
        let tokens = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::from_data(
            TensorData::new(vec![1i32, 2, 3], [1, 3]),
            &dev,
        );
        let mask = build_causal_mask::<B>(3, 3, &dev);
        let out = transformer.forward(tokens, &rope, Some(mask), &mut caches, 0);
        assert_eq!(out.dims(), [1, 3, 100]);
    }

    #[test]
    fn quantize_q8s_known_values() {
        // 32-element block with known max
        let mut data = vec![0.0f32; 32];
        data[0] = 127.0;
        data[1] = -63.5;
        data[2] = 0.0;

        let (vals, scales) = quantize_q8s(&data, 32);
        assert_eq!(vals.len(), 32);
        assert_eq!(scales.len(), 1);
        // scale = 127.0 / 127.0 = 1.0
        assert!((scales[0] - 1.0).abs() < 1e-6);
        assert_eq!(vals[0], 127);
        assert_eq!(vals[1], -64); // round(-63.5) = -64
        assert_eq!(vals[2], 0);
    }

    #[test]
    fn quantize_q8s_zero_block() {
        let data = vec![0.0f32; 32];
        let (vals, scales) = quantize_q8s(&data, 32);
        assert_eq!(scales.len(), 1);
        // scale = 1.0 (fallback for zero block)
        assert!((scales[0] - 1.0).abs() < 1e-6);
        for &v in &vals {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn quantize_q8s_multiple_blocks() {
        let mut data = vec![0.0f32; 64];
        data[0] = 25.4; // block 0 max
        data[32] = -50.8; // block 1 max
        let (vals, scales) = quantize_q8s(&data, 32);
        assert_eq!(vals.len(), 64);
        assert_eq!(scales.len(), 2);
        assert!((scales[0] - 25.4 / 127.0).abs() < 1e-5);
        assert!((scales[1] - 50.8 / 127.0).abs() < 1e-5);
    }

    #[test]
    fn quantize_q4s_known_values() {
        let mut data = vec![0.0f32; 32];
        data[0] = 7.0;
        data[1] = -3.5;
        data[2] = 0.0;

        let (vals, scales) = quantize_q4s(&data, 32);
        assert_eq!(vals.len(), 32);
        assert_eq!(scales.len(), 1);
        // scale = 7.0 / 7.0 = 1.0
        assert!((scales[0] - 1.0).abs() < 1e-6);
        assert_eq!(vals[0], 7);
        assert_eq!(vals[1], -4); // round(-3.5) = -4
        assert_eq!(vals[2], 0);
    }

    #[test]
    fn quantize_q4s_clamps_to_range() {
        let mut data = vec![0.0f32; 32];
        data[0] = 14.0; // scale = 14/7 = 2.0, val = 14/2 = 7 (at range limit)
        data[1] = -14.0; // val = -14/2 = -7 (at range limit)

        let (vals, scales) = quantize_q4s(&data, 32);
        assert!((scales[0] - 2.0).abs() < 1e-6);
        assert_eq!(vals[0], 7);
        assert_eq!(vals[1], -7);
    }

    #[test]
    fn detect_gguf_quantization_q8_0() {
        use std::collections::HashMap;
        let mut tensors = HashMap::new();
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            crate::gguf::GgufTensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dims: vec![1024, 1024],
                dtype: crate::gguf::GgufDtype::Q8_0,
                offset: 0,
                num_elements: 1024 * 1024,
                data_size: 0,
            },
        );
        let gguf = crate::gguf::GgufFile {
            metadata: HashMap::new(),
            tensors,
            tensor_data_offset: 0,
        };
        assert_eq!(detect_gguf_quantization(&gguf), QuantizationMode::Int8);
    }

    #[test]
    fn detect_gguf_quantization_q4_0() {
        use std::collections::HashMap;
        let mut tensors = HashMap::new();
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            crate::gguf::GgufTensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dims: vec![1024, 1024],
                dtype: crate::gguf::GgufDtype::Q4_0,
                offset: 0,
                num_elements: 1024 * 1024,
                data_size: 0,
            },
        );
        let gguf = crate::gguf::GgufFile {
            metadata: HashMap::new(),
            tensors,
            tensor_data_offset: 0,
        };
        assert_eq!(detect_gguf_quantization(&gguf), QuantizationMode::Int4);
    }

    #[test]
    fn detect_gguf_quantization_q4_k() {
        use std::collections::HashMap;
        let mut tensors = HashMap::new();
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            crate::gguf::GgufTensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dims: vec![1024, 1024],
                dtype: crate::gguf::GgufDtype::Q4_K,
                offset: 0,
                num_elements: 1024 * 1024,
                data_size: 0,
            },
        );
        let gguf = crate::gguf::GgufFile {
            metadata: HashMap::new(),
            tensors,
            tensor_data_offset: 0,
        };
        assert_eq!(detect_gguf_quantization(&gguf), QuantizationMode::Int4);
    }

    #[test]
    fn detect_gguf_quantization_q6_k() {
        use std::collections::HashMap;
        let mut tensors = HashMap::new();
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            crate::gguf::GgufTensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dims: vec![1024, 1024],
                dtype: crate::gguf::GgufDtype::Q6_K,
                offset: 0,
                num_elements: 1024 * 1024,
                data_size: 0,
            },
        );
        let gguf = crate::gguf::GgufFile {
            metadata: HashMap::new(),
            tensors,
            tensor_data_offset: 0,
        };
        assert_eq!(detect_gguf_quantization(&gguf), QuantizationMode::Int8);
    }

    #[test]
    fn detect_gguf_quantization_q5_k() {
        use std::collections::HashMap;
        let mut tensors = HashMap::new();
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            crate::gguf::GgufTensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dims: vec![1024, 1024],
                dtype: crate::gguf::GgufDtype::Q5_K,
                offset: 0,
                num_elements: 1024 * 1024,
                data_size: 0,
            },
        );
        let gguf = crate::gguf::GgufFile {
            metadata: HashMap::new(),
            tensors,
            tensor_data_offset: 0,
        };
        assert_eq!(detect_gguf_quantization(&gguf), QuantizationMode::Int8);
    }

    #[test]
    fn detect_gguf_quantization_q2_k() {
        use std::collections::HashMap;
        let mut tensors = HashMap::new();
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            crate::gguf::GgufTensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dims: vec![1024, 1024],
                dtype: crate::gguf::GgufDtype::Q2_K,
                offset: 0,
                num_elements: 1024 * 1024,
                data_size: 0,
            },
        );
        let gguf = crate::gguf::GgufFile {
            metadata: HashMap::new(),
            tensors,
            tensor_data_offset: 0,
        };
        assert_eq!(detect_gguf_quantization(&gguf), QuantizationMode::Int4);
    }

    #[test]
    fn detect_gguf_quantization_q3_k() {
        use std::collections::HashMap;
        let mut tensors = HashMap::new();
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            crate::gguf::GgufTensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dims: vec![1024, 1024],
                dtype: crate::gguf::GgufDtype::Q3_K,
                offset: 0,
                num_elements: 1024 * 1024,
                data_size: 0,
            },
        );
        let gguf = crate::gguf::GgufFile {
            metadata: HashMap::new(),
            tensors,
            tensor_data_offset: 0,
        };
        assert_eq!(detect_gguf_quantization(&gguf), QuantizationMode::Int4);
    }

    #[test]
    fn detect_gguf_quantization_f16_is_none() {
        use std::collections::HashMap;
        let mut tensors = HashMap::new();
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            crate::gguf::GgufTensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dims: vec![1024, 1024],
                dtype: crate::gguf::GgufDtype::F16,
                offset: 0,
                num_elements: 1024 * 1024,
                data_size: 0,
            },
        );
        let gguf = crate::gguf::GgufFile {
            metadata: HashMap::new(),
            tensors,
            tensor_data_offset: 0,
        };
        assert_eq!(detect_gguf_quantization(&gguf), QuantizationMode::None);
    }

    #[test]
    fn detect_gguf_quantization_empty_is_none() {
        use std::collections::HashMap;
        let gguf = crate::gguf::GgufFile {
            metadata: HashMap::new(),
            tensors: HashMap::new(),
            tensor_data_offset: 0,
        };
        assert_eq!(detect_gguf_quantization(&gguf), QuantizationMode::None);
    }

    #[test]
    fn fp8_e4m3_zero() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert!(fp8_e4m3_to_f32(0x80).is_sign_negative()); // -0.0
        assert_eq!(fp8_e4m3_to_f32(0x80).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn fp8_e4m3_nan() {
        assert!(fp8_e4m3_to_f32(0x7F).is_nan()); // 0_1111_111
        assert!(fp8_e4m3_to_f32(0xFF).is_nan()); // 1_1111_111
    }

    #[test]
    fn fp8_e4m3_one() {
        // 1.0 = 0_0111_000 (exp=7, mantissa=0 => 2^(7-7)*(1+0) = 1.0)
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0);
        // -1.0 = 1_0111_000
        assert_eq!(fp8_e4m3_to_f32(0xB8), -1.0);
    }

    #[test]
    fn fp8_e4m3_max() {
        // Max normal: 0_1111_110 => 2^(15-7) * (1 + 6/8) = 256 * 1.75 = 448
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0);
        assert_eq!(fp8_e4m3_to_f32(0xFE), -448.0);
    }

    #[test]
    fn fp8_e4m3_smallest_normal() {
        // Smallest normal: 0_0001_000 => 2^(1-7) * 1.0 = 2^-6
        assert_eq!(fp8_e4m3_to_f32(0x08), f32::powi(2.0, -6));
    }

    #[test]
    fn fp8_e4m3_subnormals() {
        // Subnormal: 0_0000_001 => 2^(1-7) * (1/8) = 2^-9
        assert_eq!(fp8_e4m3_to_f32(0x01), f32::powi(2.0, -9));
        // Subnormal: 0_0000_100 => 2^(1-7) * (4/8) = 2^-7
        assert_eq!(fp8_e4m3_to_f32(0x04), f32::powi(2.0, -7));
        // Largest subnormal: 0_0000_111 => 2^(1-7) * (7/8) = 7 * 2^-9
        let expected = 7.0 * f32::powi(2.0, -9);
        assert!((fp8_e4m3_to_f32(0x07) - expected).abs() < 1e-10);
    }

    #[test]
    fn fp8_e4m3_common_values() {
        // 0.5 = 0_0110_000 => 2^(6-7) = 0.5
        assert_eq!(fp8_e4m3_to_f32(0x30), 0.5);
        // 2.0 = 0_1000_000 => 2^(8-7) = 2.0
        assert_eq!(fp8_e4m3_to_f32(0x40), 2.0);
        // 1.5 = 0_0111_100 => 2^0 * (1 + 4/8) = 1.5
        assert_eq!(fp8_e4m3_to_f32(0x3C), 1.5);
        // 0.125 = 0_0100_000 => 2^(4-7) = 0.125
        assert_eq!(fp8_e4m3_to_f32(0x20), 0.125);
    }
}
