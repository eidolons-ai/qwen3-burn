use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::Path;
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::TensorData;
use serde::Deserialize;

use crate::gguf;
use crate::model::{
    apply_quantization, detect_gguf_quantization, load_gguf_weights, GenerationEvent,
    GenerationOutput, QuantizationMode, StopReason,
};
use crate::mrope::{self, MRopeEmbedding};
use crate::sampling::Sampler;
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::{build_causal_mask, AttentionKvCache, Transformer};
use crate::vision::{VisionConfig, VisionEncoder};

/// Qwen3-VL text decoder configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3VLTextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_head_dim_vl")]
    pub head_dim: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_rope_theta_vl")]
    pub rope_theta: f64,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_head_dim_vl() -> usize {
    64
}
fn default_rope_theta_vl() -> f64 {
    5_000_000.0
}

/// Qwen3-VL composite model configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3VLConfig {
    pub text_config: Qwen3VLTextConfig,
    pub vision_config: VisionConfig,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    #[serde(default = "default_video_token_id")]
    pub video_token_id: u32,
    #[serde(default = "default_vision_start_token_id")]
    pub vision_start_token_id: u32,
    #[serde(default = "default_vision_end_token_id")]
    pub vision_end_token_id: u32,
}

fn default_image_token_id() -> u32 {
    151655
}
fn default_video_token_id() -> u32 {
    151656
}
fn default_vision_start_token_id() -> u32 {
    151652
}
fn default_vision_end_token_id() -> u32 {
    151653
}

impl Qwen3VLConfig {
    /// Load from a config.json that has the VL format.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// Qwen3-VL-2B-Instruct preset.
    pub fn qwen3_vl_2b() -> Self {
        Self {
            text_config: Qwen3VLTextConfig {
                hidden_size: 2048,
                num_hidden_layers: 28,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                intermediate_size: 8192,
                vocab_size: 151936,
                rms_norm_eps: 1e-6,
                head_dim: 64,
                tie_word_embeddings: true,
                rope_theta: 5_000_000.0,
                mrope_section: Some(vec![24, 20, 20]),
            },
            vision_config: VisionConfig::qwen3_vl_2b(),
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
        }
    }
}

/// Preprocessed vision input for the VL model.
///
/// This is a backend-independent representation. Use `ImageProcessor` (with the `vision` feature)
/// to create these from image files.
pub struct VisionInput {
    /// Flattened pixel patches (normalized to [-1, 1]).
    pub pixel_patches: Vec<f32>,
    /// Grid dimensions (temporal, height, width) in patch units.
    pub grid_thw: (usize, usize, usize),
    /// Number of tokens after spatial merge.
    pub num_merge_tokens: usize,
    /// Number of patches before merge.
    pub num_patches: usize,
    /// Patch embedding dimension.
    pub patch_embed_dim: usize,
    /// Whether this input is from a video (uses `<|video_pad|>` token) vs image (`<|image_pad|>`).
    pub is_video: bool,
}

/// Generation parameters for vision-language inference.
pub struct VLGenerationParams<'a> {
    pub prompt: &'a str,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub sampler: &'a mut Sampler,
    pub prefill_chunk_size: Option<usize>,
}

/// Qwen3-VL vision-language model.
pub struct Qwen3VL<B: Backend> {
    pub(crate) transformer: Transformer<B>,
    pub(crate) vision_encoder: VisionEncoder<B>,
    pub(crate) mrope: MRopeEmbedding<B>,
    pub(crate) caches: Vec<AttentionKvCache<B>>,
    pub(crate) config: Qwen3VLConfig,
    pub(crate) max_seq_len: usize,
    pub(crate) device: Device<B>,
}

impl<B: Backend> Qwen3VL<B> {
    /// Load a Qwen3-VL model from a directory containing config.json, tokenizer.json,
    /// and SafeTensors weight files.
    pub fn from_pretrained(
        model_dir: impl AsRef<Path>,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config = Qwen3VLConfig::from_file(model_dir.join("config.json"))?;
        let tc = &config.text_config;
        let vc = &config.vision_config;

        eprintln!(
            "Qwen3-VL config: text hidden={}, layers={}, heads={}, kv_heads={}, vocab={}",
            tc.hidden_size,
            tc.num_hidden_layers,
            tc.num_attention_heads,
            tc.num_key_value_heads,
            tc.vocab_size,
        );
        eprintln!(
            "  vision: hidden={}, depth={}, heads={}, out_hidden={}",
            vc.hidden_size, vc.depth, vc.num_heads, vc.out_hidden_size,
        );

        // Build text transformer
        let transformer = Transformer::new(
            tc.vocab_size,
            tc.hidden_size,
            tc.num_hidden_layers,
            tc.num_attention_heads,
            tc.num_key_value_heads,
            tc.head_dim,
            tc.intermediate_size,
            tc.rms_norm_eps,
            tc.tie_word_embeddings,
            None, // no MoE in VL models
            device,
        );

        // Build vision encoder
        let vision_encoder = VisionEncoder::new(vc, device);

        // Build mRoPE
        let mrope_section = tc.mrope_section.clone().unwrap_or_else(|| vec![24, 20, 20]);
        let mrope = MRopeEmbedding::new(
            [mrope_section[0], mrope_section[1], mrope_section[2]],
            tc.rope_theta,
            tc.head_dim,
            device,
        );

        // Load SafeTensors weights
        let (transformer, vision_encoder) =
            load_vl_safetensors_weights(transformer, vision_encoder, model_dir, &config, device)?;

        let caches = (0..tc.num_hidden_layers)
            .map(|_| {
                AttentionKvCache::new(1, tc.num_key_value_heads, max_seq_len, tc.head_dim, device)
            })
            .collect();

        Ok(Self {
            transformer,
            vision_encoder,
            mrope,
            caches,
            config,
            max_seq_len,
            device: device.clone(),
        })
    }

    /// Load a Qwen3-VL model from a GGUF file (text decoder) plus an mmproj GGUF
    /// (vision encoder) in the same directory.
    ///
    /// The mmproj file is auto-discovered by scanning for `mmproj-*.gguf` in the
    /// same directory as the main GGUF file.
    pub fn from_gguf(
        gguf_path: impl AsRef<Path>,
        max_seq_len: usize,
        quantization: QuantizationMode,
        device: &Device<B>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let gguf_path = gguf_path.as_ref();

        // 1. Parse main GGUF (text decoder)
        let (gguf_file, mut file) = gguf::GgufFile::open(gguf_path)?;

        // 2. Extract text config from GGUF metadata
        let text_config_raw = gguf::extract_config(&gguf_file)?;
        eprintln!(
            "GGUF VL text config: hidden={}, layers={}, heads={}, kv_heads={}, vocab={}",
            text_config_raw.hidden_size,
            text_config_raw.num_hidden_layers,
            text_config_raw.num_attention_heads,
            text_config_raw.num_key_value_heads,
            text_config_raw.vocab_size,
        );

        // 3. Find mmproj file in the same directory
        let gguf_dir = gguf_path.parent().unwrap_or_else(|| Path::new("."));
        let mmproj_path = find_mmproj_gguf(gguf_dir)?;
        eprintln!("Found mmproj: {}", mmproj_path.display());

        // 4. Parse mmproj GGUF
        let (mmproj_gguf, mut mmproj_file) = gguf::GgufFile::open(&mmproj_path)?;

        // Log mmproj tensor names for debugging
        let mut mmproj_names: Vec<&String> = mmproj_gguf.tensors.keys().collect();
        mmproj_names.sort();
        eprintln!("mmproj tensors ({}):", mmproj_names.len());
        for name in &mmproj_names {
            eprintln!("  {}", name);
        }

        // 5. Extract vision config from mmproj metadata
        let vision_config = gguf::extract_vision_config(&mmproj_gguf, text_config_raw.hidden_size)?;
        eprintln!(
            "  vision: hidden={}, depth={}, heads={}, out_hidden={}, patch={}",
            vision_config.hidden_size,
            vision_config.depth,
            vision_config.num_heads,
            vision_config.out_hidden_size,
            vision_config.patch_size,
        );
        if let Some(ref ds) = vision_config.deepstack_visual_indexes {
            eprintln!("  deepstack indexes: {:?}", ds);
        }

        // 6. Read head_dim and rope_theta from GGUF metadata for VL text config
        let arch = gguf_file
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("qwen3vl")
            .to_string();
        let head_dim = gguf_file
            .metadata
            .get(&format!("{}.attention.key_length", arch))
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;
        let rope_theta = gguf_file
            .metadata
            .get(&format!("{}.rope.freq_base", arch))
            .and_then(|v| v.as_f64())
            .unwrap_or(5_000_000.0);

        // Try to read mrope_section from GGUF metadata
        let mrope_section = gguf_file
            .metadata
            .get(&format!("{}.rope.dimension_sections", arch))
            .and_then(|v| {
                if let gguf::MetadataValue::Array(arr) = v {
                    let vals: Vec<usize> = arr
                        .iter()
                        .filter_map(|e| e.as_u64().map(|x| x as usize))
                        .collect();
                    if vals.len() >= 3 {
                        Some(vals)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                // Compute from head_dim: for head_dim=64 -> [24, 20, 20]
                // Total must equal head_dim. Section 0 is slightly larger.
                let s2 = head_dim / 3; // 20 for 64, 21 for 64
                let s0 = head_dim - 2 * s2;
                vec![s0, s2, s2]
            });

        let tc = Qwen3VLTextConfig {
            hidden_size: text_config_raw.hidden_size,
            num_hidden_layers: text_config_raw.num_hidden_layers,
            num_attention_heads: text_config_raw.num_attention_heads,
            num_key_value_heads: text_config_raw.num_key_value_heads,
            intermediate_size: text_config_raw.intermediate_size,
            vocab_size: text_config_raw.vocab_size,
            rms_norm_eps: text_config_raw.rms_norm_eps,
            head_dim,
            tie_word_embeddings: text_config_raw.tie_word_embeddings,
            rope_theta,
            mrope_section: Some(mrope_section.clone()),
        };

        let config = Qwen3VLConfig {
            text_config: tc,
            vision_config: vision_config.clone(),
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
        };

        // 7. Create text transformer skeleton
        let transformer = Transformer::new(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            config.text_config.num_hidden_layers,
            config.text_config.num_attention_heads,
            config.text_config.num_key_value_heads,
            config.text_config.head_dim,
            config.text_config.intermediate_size,
            config.text_config.rms_norm_eps,
            config.text_config.tie_word_embeddings,
            None, // no MoE in VL models
            device,
        );

        // 8. Resolve quantization
        let detected = detect_gguf_quantization(&gguf_file);
        let resolved_quant = match quantization {
            QuantizationMode::Auto => detected,
            other => other,
        };

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

        // 9. Load text weights
        let per_tensor_quantized = quant_scheme.is_some();
        let transformer = load_gguf_weights(
            transformer,
            &gguf_file,
            &mut file,
            &text_config_raw,
            quant_scheme,
            device,
        )?;

        let transformer = if per_tensor_quantized {
            transformer
        } else {
            apply_quantization(transformer, resolved_quant)
        };

        // 10. Create vision encoder skeleton and load weights
        let vision_encoder = VisionEncoder::new(&vision_config, device);
        let vision_encoder =
            load_gguf_vision_weights(vision_encoder, &mmproj_gguf, &mut mmproj_file, device)?;

        // 11. Build mRoPE
        let mrope = MRopeEmbedding::new(
            [mrope_section[0], mrope_section[1], mrope_section[2]],
            rope_theta,
            head_dim,
            device,
        );

        // 12. Create KV caches
        let caches = (0..config.text_config.num_hidden_layers)
            .map(|_| {
                AttentionKvCache::new(
                    1,
                    config.text_config.num_key_value_heads,
                    max_seq_len,
                    config.text_config.head_dim,
                    device,
                )
            })
            .collect();

        Ok(Self {
            transformer,
            vision_encoder,
            mrope,
            caches,
            config,
            max_seq_len,
            device: device.clone(),
        })
    }

    /// Reset KV caches.
    pub fn reset_caches(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    /// Generate text with vision inputs.
    ///
    /// - `tokenizer`: the tokenizer
    /// - `token_ids`: pre-tokenized prompt (with image_pad tokens already inserted)
    /// - `images`: preprocessed vision inputs (patches + grid info)
    /// - `params`: generation parameters
    /// - `callback`: streaming callback
    pub fn generate_with_vision(
        &mut self,
        tokenizer: &Qwen3Tokenizer,
        token_ids: &[u32],
        images: &[VisionInput],
        params: VLGenerationParams,
        mut callback: impl FnMut(GenerationEvent) -> ControlFlow<()>,
    ) -> Result<GenerationOutput, String> {
        self.reset_caches();

        let prompt_len = token_ids.len();
        if prompt_len > self.max_seq_len {
            return Err(format!(
                "prompt length ({}) exceeds max_seq_len ({})",
                prompt_len, self.max_seq_len
            ));
        }

        let tc = &self.config.text_config;
        let start_time = Instant::now();

        // 1. Run vision encoder on each image
        let mut vision_outputs = Vec::new();
        for img in images {
            let (grid_t, grid_h, grid_w) = img.grid_thw;
            let patches = Tensor::<B, 1>::from_floats(&img.pixel_patches[..], &self.device)
                .reshape([img.num_patches, img.patch_embed_dim]);

            let output = self.vision_encoder.forward(patches, grid_t, grid_h, grid_w);

            vision_outputs.push((output, img.grid_thw));
        }

        // 2. Build input embeddings: start with text embeddings
        let token_data: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        let token_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(token_data, [prompt_len]), &self.device)
                .unsqueeze::<2>(); // [1, seq_len]

        let mut embeddings = self.transformer.embed_tokens().forward(token_tensor);
        // embeddings: [1, seq_len, hidden_size]

        // 3. Replace vision token positions with vision features
        let image_token_id = self.config.image_token_id;
        let video_token_id = self.config.video_token_id;

        // Build visual mask for DeepStack: [1, seq_len, 1]
        let mut visual_mask_data = vec![0.0f32; prompt_len];

        let mut scan_pos = 0; // cumulative scan position across all vision inputs
        for (img_idx, (vo, _grid_thw)) in vision_outputs.iter().enumerate() {
            // Determine which pad token this input uses
            let target_token_id = if images[img_idx].is_video {
                video_token_id
            } else {
                image_token_id
            };

            let pooler = &vo.pooler_output; // [num_merge_tokens, out_hidden_size]
            let [num_merge, out_hidden] = pooler.dims();

            // Find this input's pad token span (continuing from previous input's end)
            let mut replaced = 0;
            while scan_pos < prompt_len && replaced < num_merge {
                if token_ids[scan_pos] == target_token_id {
                    let feat = pooler
                        .clone()
                        .slice([replaced..replaced + 1, 0..out_hidden])
                        .unsqueeze_dim::<3>(0); // [1, 1, hidden]

                    embeddings = embeddings
                        .slice_assign([0..1, scan_pos..scan_pos + 1, 0..tc.hidden_size], feat);
                    visual_mask_data[scan_pos] = 1.0;
                    replaced += 1;
                }
                scan_pos += 1;
            }
        }

        // 4. Build 3D position IDs (use merged grid dimensions, not pre-merge)
        let merge = self.config.vision_config.spatial_merge_size;
        let merged_grid_thws: Vec<(usize, usize, usize)> = vision_outputs
            .iter()
            .map(|(_, g)| (g.0, g.1 / merge, g.2 / merge))
            .collect();
        let position_ids = mrope::build_position_ids(
            token_ids,
            &[image_token_id, video_token_id],
            &merged_grid_thws,
        );

        // 5. Compute mRoPE cos/sin
        let (cos, sin) = self.mrope.compute_cos_sin(&position_ids, &self.device);

        // 6. Build DeepStack features: scatter vision features to vision token positions
        let vision_positions: Vec<usize> = token_ids
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == image_token_id || t == video_token_id)
            .map(|(i, _)| i)
            .collect();

        let num_ds = self.vision_encoder.deepstack_visual_indexes.len();
        let deepstack_layers: Vec<usize> = (0..num_ds).collect();
        let mut deepstack_features: Vec<Tensor<B, 3>> = Vec::with_capacity(num_ds);

        // For each deepstack index, collect features from all images
        for ds_idx in 0..num_ds {
            let mut full = Tensor::<B, 3>::zeros([1, prompt_len, tc.hidden_size], &self.device);
            let mut img_offset = 0;

            for (vo, _) in &vision_outputs {
                if ds_idx < vo.deepstack_features.len() {
                    let feat = &vo.deepstack_features[ds_idx]; // [num_merge, out_hidden]
                    let [num_merge, out_hidden] = feat.dims();

                    for feat_idx in 0..num_merge {
                        let global_pos = img_offset + feat_idx;
                        if global_pos >= vision_positions.len() {
                            break;
                        }
                        let pos = vision_positions[global_pos];
                        let f = feat
                            .clone()
                            .slice([feat_idx..feat_idx + 1, 0..out_hidden])
                            .unsqueeze_dim::<3>(0); // [1, 1, hidden]
                        full = full.slice_assign([0..1, pos..pos + 1, 0..tc.hidden_size], f);
                    }
                    img_offset += vo.pooler_output.dims()[0];
                }
            }

            deepstack_features.push(full);
        }

        let visual_mask_tensor = Tensor::<B, 1>::from_floats(&visual_mask_data[..], &self.device)
            .reshape([1, prompt_len, 1])
            .expand([1, prompt_len, tc.hidden_size]);

        // 8. Prefill (optionally in chunks)
        let chunk_size = params.prefill_chunk_size.unwrap_or(prompt_len);
        let num_chunks = prompt_len.div_ceil(chunk_size);
        let has_deepstack = !deepstack_features.is_empty();
        let mut last_logits_2d: Option<Tensor<B, 2>> = None;
        let mut cancelled = false;
        let mut chunk_start = 0;

        for chunk_idx in 0..num_chunks {
            let chunk_end = (chunk_start + chunk_size).min(prompt_len);
            let chunk_len = chunk_end - chunk_start;
            let total_seq_len = chunk_end; // KV cache has accumulated this many positions

            let emb_chunk =
                embeddings
                    .clone()
                    .slice([0..1, chunk_start..chunk_end, 0..tc.hidden_size]);
            let cos_chunk = cos.clone().slice([chunk_start..chunk_end, 0..tc.head_dim]);
            let sin_chunk = sin.clone().slice([chunk_start..chunk_end, 0..tc.head_dim]);
            let mask_chunk = build_causal_mask::<B>(chunk_len, total_seq_len, &self.device);

            let logits = if has_deepstack {
                let ds_chunks: Vec<Tensor<B, 3>> = deepstack_features
                    .iter()
                    .map(|f| {
                        f.clone()
                            .slice([0..1, chunk_start..chunk_end, 0..tc.hidden_size])
                    })
                    .collect();
                let vm_chunk = visual_mask_tensor.clone().slice([
                    0..1,
                    chunk_start..chunk_end,
                    0..tc.hidden_size,
                ]);

                self.transformer.forward_from_embeddings(
                    emb_chunk,
                    cos_chunk,
                    sin_chunk,
                    Some(mask_chunk),
                    &mut self.caches,
                    Some((&ds_chunks, vm_chunk, &deepstack_layers)),
                )
            } else {
                self.transformer.forward_from_embeddings(
                    emb_chunk,
                    cos_chunk,
                    sin_chunk,
                    Some(mask_chunk),
                    &mut self.caches,
                    None,
                )
            };

            let chunk_last_logits = logits
                .slice([0..1, (chunk_len - 1)..chunk_len, 0..tc.vocab_size])
                .reshape([1, tc.vocab_size]);
            last_logits_2d = Some(chunk_last_logits);

            chunk_start = chunk_end;

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
        let eos_token = tokenizer.eos_token_id();
        let mut all_tokens: Vec<u32> = token_ids.to_vec();
        all_tokens.push(next_token);
        let mut generated_count = 1;

        let stop_reason = if next_token == eos_token {
            StopReason::Eos
        } else {
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

            // 9. Autoregressive decode
            let mut pos = prompt_len;
            let mut text_pos = position_ids[0].last().copied().unwrap_or(0) + 1;
            let mut stop = StopReason::MaxTokens;

            for _ in 1..params.max_new_tokens {
                let td = TensorData::new(vec![next_token as i32], [1]);
                let token_tensor =
                    Tensor::<B, 1, Int>::from_data(td, &self.device).unsqueeze::<2>();
                let token_emb = self.transformer.embed_tokens().forward(token_tensor);

                // Text-only decode: all 3 mRoPE dims identical
                let decode_pos_ids = vec![vec![text_pos], vec![text_pos], vec![text_pos]];
                let (cos, sin) = self.mrope.compute_cos_sin(&decode_pos_ids, &self.device);

                let logits = self.transformer.forward_from_embeddings(
                    token_emb,
                    cos,
                    sin,
                    None, // no mask for single-token decode
                    &mut self.caches,
                    None, // no deepstack during decode
                );

                let logits = logits.reshape([1, tc.vocab_size]);
                next_token = sample_token(&logits, params.temperature, params.sampler);
                all_tokens.push(next_token);
                generated_count += 1;
                pos += 1;
                text_pos += 1;

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

/// Apply temperature scaling and sample a token (same as model.rs).
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

    let max = logits_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits_f64
        .iter()
        .map(|&x| ((x - max) / temperature).exp())
        .collect();
    let sum: f64 = exp.iter().sum();
    let probs: Vec<f64> = exp.iter().map(|&x| x / sum).collect();

    sampler.sample_probs(&probs)
}

/// Find an mmproj GGUF file in the given directory.
///
/// Scans for files matching `mmproj-*.gguf` or `*mmproj*.gguf`.
fn find_mmproj_gguf(dir: &Path) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.ends_with(".gguf") && name.contains("mmproj") {
                candidates.push(path);
            }
        }
    }
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| format!("No mmproj-*.gguf file found in {}", dir.display()).into())
}

/// Load vision encoder weights from an mmproj GGUF file.
///
/// All vision weights are loaded as f32 (no quantization). The GGUF parser
/// dequantizes F16/Q8_0 to f32 automatically. 2D weights (linear layers)
/// are transposed from `[out, in]` to `[in, out]`.
fn load_gguf_vision_weights<B: Backend>(
    mut vision_encoder: VisionEncoder<B>,
    gguf_file: &gguf::GgufFile,
    file: &mut std::fs::File,
    device: &Device<B>,
) -> Result<VisionEncoder<B>, Box<dyn std::error::Error>> {
    // Helper: read a tensor, return f32 data with reversed dims (PyTorch convention)
    let read_tensor = |file: &mut std::fs::File,
                       name: &str|
     -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
        let info = gguf_file
            .tensors
            .get(name)
            .ok_or_else(|| format!("mmproj tensor '{}' not found", name))?;
        let data = gguf_file.read_tensor_data(file, info)?;
        let shape: Vec<usize> = info.dims.iter().rev().copied().collect();
        Ok((data, shape))
    };

    let make_1d = |data: Vec<f32>, device: &Device<B>| -> Tensor<B, 1> {
        let len = data.len();
        Tensor::from_data(TensorData::new(data, [len]), device)
    };

    let make_2d = |data: Vec<f32>, shape: &[usize], device: &Device<B>| -> Tensor<B, 2> {
        Tensor::from_data(TensorData::new(data, [shape[0], shape[1]]), device)
    };

    // 2D linear weight: transpose [out, in] -> [in, out]
    let make_2d_linear = |data: Vec<f32>, shape: &[usize], device: &Device<B>| -> Tensor<B, 2> {
        let t = Tensor::from_data(TensorData::new(data, [shape[0], shape[1]]), device);
        t.transpose()
    };

    // Track which tensors we've consumed for warning about unrecognized ones
    let mut consumed = std::collections::HashSet::new();

    // --- Patch embedding ---
    // GGUF splits the Conv3D temporal dimension into separate tensors:
    //   v.patch_embd.weight   → temporal slice 0: [out_ch, in_ch, patch_h, patch_w]
    //   v.patch_embd.weight.1 → temporal slice 1: [out_ch, in_ch, patch_h, patch_w]
    // We need to concatenate along the spatial dim to reconstruct
    // [out_ch, in_ch * temporal * patch_h * patch_w].
    if gguf_file.tensors.contains_key("v.patch_embd.weight") {
        let (data0, shape0) = read_tensor(file, "v.patch_embd.weight")?;
        consumed.insert("v.patch_embd.weight".to_string());
        let out_dim = shape0[0];
        let slice_dim: usize = shape0[1..].iter().product();

        let weight = if gguf_file.tensors.contains_key("v.patch_embd.weight.1") {
            let (data1, _) = read_tensor(file, "v.patch_embd.weight.1")?;
            consumed.insert("v.patch_embd.weight.1".to_string());
            // Concatenate: each slice is [out_dim, slice_dim], result is [out_dim, 2*slice_dim]
            let mut combined = Vec::with_capacity(data0.len() + data1.len());
            // Interleave rows: for each output channel, concat slice0 row then slice1 row
            for row in 0..out_dim {
                let start = row * slice_dim;
                combined.extend_from_slice(&data0[start..start + slice_dim]);
                combined.extend_from_slice(&data1[start..start + slice_dim]);
            }
            let total_dim = 2 * slice_dim;
            Tensor::<B, 1>::from_data(TensorData::new(combined, [out_dim * total_dim]), device)
                .reshape([out_dim, total_dim])
        } else {
            Tensor::<B, 1>::from_data(TensorData::new(data0, [out_dim * slice_dim]), device)
                .reshape([out_dim, slice_dim])
        };

        let (bias_data, _) = read_tensor(file, "v.patch_embd.bias")?;
        consumed.insert("v.patch_embd.bias".to_string());
        let bias = make_1d(bias_data, device);
        vision_encoder = vision_encoder.load_patch_embed(weight, bias);
    }

    // --- Position embeddings ---
    if gguf_file.tensors.contains_key("v.position_embd.weight") {
        let (data, shape) = read_tensor(file, "v.position_embd.weight")?;
        consumed.insert("v.position_embd.weight".to_string());
        let pos_weight = make_2d(data, &shape, device);
        vision_encoder = vision_encoder.load_pos_embed(pos_weight);
    }

    // --- Vision blocks ---
    let depth = vision_encoder.blocks.len();
    for i in 0..depth {
        let prefix = format!("v.blk.{}", i);

        // Check if this block exists in the GGUF
        let ln1_w_key = format!("{}.ln1.weight", prefix);
        if !gguf_file.tensors.contains_key(&ln1_w_key) {
            continue;
        }

        let load_1d = |file: &mut std::fs::File,
                       key: &str,
                       consumed: &mut std::collections::HashSet<String>|
         -> Result<Tensor<B, 1>, Box<dyn std::error::Error>> {
            let (data, _) = read_tensor(file, key)?;
            consumed.insert(key.to_string());
            Ok(make_1d(data, device))
        };

        let load_2d_linear = |file: &mut std::fs::File,
                              key: &str,
                              consumed: &mut std::collections::HashSet<String>|
         -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
            let (data, shape) = read_tensor(file, key)?;
            consumed.insert(key.to_string());
            Ok(make_2d_linear(data, &shape, device))
        };

        let norm1_w = load_1d(file, &format!("{}.ln1.weight", prefix), &mut consumed)?;
        let norm1_b = load_1d(file, &format!("{}.ln1.bias", prefix), &mut consumed)?;
        let norm2_w = load_1d(file, &format!("{}.ln2.weight", prefix), &mut consumed)?;
        let norm2_b = load_1d(file, &format!("{}.ln2.bias", prefix), &mut consumed)?;
        let qkv_w = load_2d_linear(file, &format!("{}.attn_qkv.weight", prefix), &mut consumed)?;
        let qkv_b = load_1d(file, &format!("{}.attn_qkv.bias", prefix), &mut consumed)?;
        let proj_w = load_2d_linear(file, &format!("{}.attn_out.weight", prefix), &mut consumed)?;
        let proj_b = load_1d(file, &format!("{}.attn_out.bias", prefix), &mut consumed)?;
        let fc1_w = load_2d_linear(file, &format!("{}.ffn_up.weight", prefix), &mut consumed)?;
        let fc1_b = load_1d(file, &format!("{}.ffn_up.bias", prefix), &mut consumed)?;
        let fc2_w = load_2d_linear(file, &format!("{}.ffn_down.weight", prefix), &mut consumed)?;
        let fc2_b = load_1d(file, &format!("{}.ffn_down.bias", prefix), &mut consumed)?;

        vision_encoder.load_block(
            i, norm1_w, norm1_b, norm2_w, norm2_b, qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b,
            fc2_w, fc2_b,
        );

        if i % 10 == 0 {
            eprintln!("Loading vision block {}/{}...", i, depth);
        }
    }

    // --- Main merger ---
    // Try mm.post_norm (standard) or v.post_ln (variant)
    let norm_key = if gguf_file.tensors.contains_key("mm.post_norm.weight") {
        "mm.post_norm"
    } else if gguf_file.tensors.contains_key("v.post_ln.weight") {
        "v.post_ln"
    } else {
        return Err("No merger norm found (tried mm.post_norm, v.post_ln)".into());
    };

    let (norm_w_data, _) = read_tensor(file, &format!("{}.weight", norm_key))?;
    consumed.insert(format!("{}.weight", norm_key));
    let norm_w = make_1d(norm_w_data, device);
    let (norm_b_data, _) = read_tensor(file, &format!("{}.bias", norm_key))?;
    consumed.insert(format!("{}.bias", norm_key));
    let norm_b = make_1d(norm_b_data, device);

    // FC layers: try mm.0/mm.1 (standard) or mm.0/mm.2 (variant)
    let fc1_key = "mm.0";
    let fc2_key = if gguf_file.tensors.contains_key("mm.1.weight") {
        "mm.1"
    } else if gguf_file.tensors.contains_key("mm.2.weight") {
        "mm.2"
    } else {
        return Err("No merger fc2 found (tried mm.1, mm.2)".into());
    };

    let (fc1_w_data, fc1_w_shape) = read_tensor(file, &format!("{}.weight", fc1_key))?;
    consumed.insert(format!("{}.weight", fc1_key));
    let fc1_w = make_2d_linear(fc1_w_data, &fc1_w_shape, device);
    let (fc1_b_data, _) = read_tensor(file, &format!("{}.bias", fc1_key))?;
    consumed.insert(format!("{}.bias", fc1_key));
    let fc1_b = make_1d(fc1_b_data, device);

    let (fc2_w_data, fc2_w_shape) = read_tensor(file, &format!("{}.weight", fc2_key))?;
    consumed.insert(format!("{}.weight", fc2_key));
    let fc2_w = make_2d_linear(fc2_w_data, &fc2_w_shape, device);
    let (fc2_b_data, _) = read_tensor(file, &format!("{}.bias", fc2_key))?;
    consumed.insert(format!("{}.bias", fc2_key));
    let fc2_b = make_1d(fc2_b_data, device);

    vision_encoder.load_merger(norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b);

    // --- Deepstack mergers ---
    // GGUF uses actual layer indices as keys (e.g., v.deepstack.5, v.deepstack.11, v.deepstack.17),
    // while VisionEncoder uses sequential merger indices (0, 1, 2).
    // Collect and sort the actual layer indices from tensor names.
    let mut ds_layer_indices: Vec<usize> = Vec::new();
    for name in gguf_file.tensors.keys() {
        if let Some(rest) = name.strip_prefix("v.deepstack.") {
            if let Some(dot_pos) = rest.find('.') {
                if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                    if !ds_layer_indices.contains(&idx) {
                        ds_layer_indices.push(idx);
                    }
                }
            }
        }
    }
    ds_layer_indices.sort();

    for (merger_idx, &layer_idx) in ds_layer_indices.iter().enumerate() {
        let ds_prefix = format!("v.deepstack.{}", layer_idx);

        let (ds_norm_w_data, _) = read_tensor(file, &format!("{}.norm.weight", ds_prefix))?;
        consumed.insert(format!("{}.norm.weight", ds_prefix));
        let ds_norm_w = make_1d(ds_norm_w_data, device);
        let (ds_norm_b_data, _) = read_tensor(file, &format!("{}.norm.bias", ds_prefix))?;
        consumed.insert(format!("{}.norm.bias", ds_prefix));
        let ds_norm_b = make_1d(ds_norm_b_data, device);

        let (ds_fc1_w_data, ds_fc1_w_shape) =
            read_tensor(file, &format!("{}.fc1.weight", ds_prefix))?;
        consumed.insert(format!("{}.fc1.weight", ds_prefix));
        let ds_fc1_w = make_2d_linear(ds_fc1_w_data, &ds_fc1_w_shape, device);
        let (ds_fc1_b_data, _) = read_tensor(file, &format!("{}.fc1.bias", ds_prefix))?;
        consumed.insert(format!("{}.fc1.bias", ds_prefix));
        let ds_fc1_b = make_1d(ds_fc1_b_data, device);

        let (ds_fc2_w_data, ds_fc2_w_shape) =
            read_tensor(file, &format!("{}.fc2.weight", ds_prefix))?;
        consumed.insert(format!("{}.fc2.weight", ds_prefix));
        let ds_fc2_w = make_2d_linear(ds_fc2_w_data, &ds_fc2_w_shape, device);
        let (ds_fc2_b_data, _) = read_tensor(file, &format!("{}.fc2.bias", ds_prefix))?;
        consumed.insert(format!("{}.fc2.bias", ds_prefix));
        let ds_fc2_b = make_1d(ds_fc2_b_data, device);

        vision_encoder.load_deepstack_merger(
            merger_idx, ds_norm_w, ds_norm_b, ds_fc1_w, ds_fc1_b, ds_fc2_w, ds_fc2_b,
        );
    }

    // Log unrecognized tensors
    let unrecognized: Vec<&String> = gguf_file
        .tensors
        .keys()
        .filter(|k| !consumed.contains(*k))
        .collect();
    if !unrecognized.is_empty() {
        eprintln!("Warning: {} mmproj tensors not loaded:", unrecognized.len());
        let mut sorted: Vec<&&String> = unrecognized.iter().collect();
        sorted.sort();
        for k in sorted.iter().take(20) {
            eprintln!("  - {}", k);
        }
        if sorted.len() > 20 {
            eprintln!("  ... and {} more", sorted.len() - 20);
        }
    }

    eprintln!("Vision encoder loaded successfully.");
    Ok(vision_encoder)
}

type TensorMap = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// Remove a 1D tensor from the map.
fn take_tensor_1d<B: Backend>(
    map: &mut TensorMap,
    name: &str,
    device: &Device<B>,
) -> Result<Tensor<B, 1>, Box<dyn std::error::Error>> {
    let (data, _shape) = map
        .remove(name)
        .ok_or_else(|| format!("Tensor '{}' not found", name))?;
    let len = data.len();
    Ok(Tensor::from_data(TensorData::new(data, [len]), device))
}

/// Remove a 2D tensor (no transpose).
fn take_tensor_2d<B: Backend>(
    map: &mut TensorMap,
    name: &str,
    device: &Device<B>,
) -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
    let (data, shape) = map
        .remove(name)
        .ok_or_else(|| format!("Tensor '{}' not found", name))?;
    Ok(Tensor::from_data(
        TensorData::new(data, [shape[0], shape[1]]),
        device,
    ))
}

/// Remove a 2D Linear weight and transpose [out, in] -> [in, out].
fn take_linear_weight<B: Backend>(
    map: &mut TensorMap,
    name: &str,
    device: &Device<B>,
) -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
    Ok(take_tensor_2d(map, name, device)?.transpose())
}

/// Check if all tensors for a text layer are available in the map.
fn vl_text_layer_ready(layer_idx: usize, map: &TensorMap) -> bool {
    let p = format!("model.language_model.layers.{}", layer_idx);
    let has = |suffix: &str| map.contains_key(&format!("{}.{}", p, suffix));

    has("self_attn.q_proj.weight")
        && has("self_attn.k_proj.weight")
        && has("self_attn.v_proj.weight")
        && has("self_attn.o_proj.weight")
        && has("self_attn.q_norm.weight")
        && has("self_attn.k_norm.weight")
        && has("mlp.gate_proj.weight")
        && has("mlp.up_proj.weight")
        && has("mlp.down_proj.weight")
        && has("input_layernorm.weight")
        && has("post_attention_layernorm.weight")
}

/// Load a single text layer's weights from the map, consuming them.
fn load_vl_text_layer<B: Backend>(
    map: &mut TensorMap,
    transformer: Transformer<B>,
    layer_idx: usize,
    device: &Device<B>,
) -> Result<Transformer<B>, Box<dyn std::error::Error>> {
    let p = format!("model.language_model.layers.{}", layer_idx);

    let q_proj_w = take_linear_weight(map, &format!("{}.self_attn.q_proj.weight", p), device)?;
    let k_proj_w = take_linear_weight(map, &format!("{}.self_attn.k_proj.weight", p), device)?;
    let v_proj_w = take_linear_weight(map, &format!("{}.self_attn.v_proj.weight", p), device)?;
    let o_proj_w = take_linear_weight(map, &format!("{}.self_attn.o_proj.weight", p), device)?;
    let q_norm_w = take_tensor_1d(map, &format!("{}.self_attn.q_norm.weight", p), device)?;
    let k_norm_w = take_tensor_1d(map, &format!("{}.self_attn.k_norm.weight", p), device)?;
    let gate_proj_w = take_linear_weight(map, &format!("{}.mlp.gate_proj.weight", p), device)?;
    let up_proj_w = take_linear_weight(map, &format!("{}.mlp.up_proj.weight", p), device)?;
    let down_proj_w = take_linear_weight(map, &format!("{}.mlp.down_proj.weight", p), device)?;
    let input_ln_w = take_tensor_1d(map, &format!("{}.input_layernorm.weight", p), device)?;
    let post_attn_ln_w = take_tensor_1d(
        map,
        &format!("{}.post_attention_layernorm.weight", p),
        device,
    )?;

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

/// Check if all tensors for a vision block are available in the map.
fn vision_block_ready(block_idx: usize, map: &TensorMap) -> bool {
    let p = format!("model.visual.blocks.{}", block_idx);
    let has = |suffix: &str| map.contains_key(&format!("{}.{}", p, suffix));

    has("norm1.weight")
        && has("norm1.bias")
        && has("norm2.weight")
        && has("norm2.bias")
        && has("attn.qkv.weight")
        && has("attn.qkv.bias")
        && has("attn.proj.weight")
        && has("attn.proj.bias")
        && has("mlp.linear_fc1.weight")
        && has("mlp.linear_fc1.bias")
        && has("mlp.linear_fc2.weight")
        && has("mlp.linear_fc2.bias")
}

/// Check if all tensors for a merger (main or deepstack) are available in the map.
fn merger_ready(prefix: &str, map: &TensorMap) -> bool {
    let has = |suffix: &str| map.contains_key(&format!("{}.{}", prefix, suffix));

    has("norm.weight")
        && has("norm.bias")
        && has("linear_fc1.weight")
        && has("linear_fc1.bias")
        && has("linear_fc2.weight")
        && has("linear_fc2.bias")
}

/// Load a merger's weights from the map, consuming them.
#[allow(clippy::type_complexity)]
fn load_merger_weights<B: Backend>(
    map: &mut TensorMap,
    prefix: &str,
    device: &Device<B>,
) -> Result<
    (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 2>,
        Tensor<B, 1>,
        Tensor<B, 2>,
        Tensor<B, 1>,
    ),
    Box<dyn std::error::Error>,
> {
    let norm_w = take_tensor_1d(map, &format!("{}.norm.weight", prefix), device)?;
    let norm_b = take_tensor_1d(map, &format!("{}.norm.bias", prefix), device)?;
    let fc1_w = take_linear_weight(map, &format!("{}.linear_fc1.weight", prefix), device)?;
    let fc1_b = take_tensor_1d(map, &format!("{}.linear_fc1.bias", prefix), device)?;
    let fc2_w = take_linear_weight(map, &format!("{}.linear_fc2.weight", prefix), device)?;
    let fc2_b = take_tensor_1d(map, &format!("{}.linear_fc2.bias", prefix), device)?;
    Ok((norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b))
}

/// Load SafeTensors weights for Qwen3-VL (text + vision).
///
/// Streams weights shard-by-shard: after reading each shard file, completed layers are
/// loaded into the model immediately and their f32 data is freed. This keeps peak memory
/// close to the final model size rather than 2-3x that.
///
/// Key prefix mapping:
/// - `model.language_model.*` -> text transformer
/// - `model.visual.*` -> vision encoder
fn load_vl_safetensors_weights<B: Backend>(
    mut transformer: Transformer<B>,
    mut vision_encoder: VisionEncoder<B>,
    model_dir: &Path,
    config: &Qwen3VLConfig,
    device: &Device<B>,
) -> Result<(Transformer<B>, VisionEncoder<B>), Box<dyn std::error::Error>> {
    let tc = &config.text_config;

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
        return Err("No .safetensors files found".into());
    }
    st_files.sort();

    let mut tensor_map: TensorMap = HashMap::new();
    let mut next_text_layer: usize = 0;
    let mut next_vision_block: usize = 0;
    let mut embed_loaded = false;
    let mut lm_head_weight: Option<Tensor<B, 2>> = None;
    let mut merger_loaded = false;
    let ds_indexes = config
        .vision_config
        .deepstack_visual_indexes
        .clone()
        .unwrap_or_default();
    let mut next_ds_merger: usize = 0;

    // Stream weights shard-by-shard: after reading each shard file, completed layers are
    // loaded into the model immediately and their f32 data is freed. This keeps peak memory
    // close to the final model size rather than 2-3x that.
    for (shard_idx, path) in st_files.iter().enumerate() {
        eprintln!("Reading shard {}/{}...", shard_idx + 1, st_files.len());

        // Read shard and extract tensors into tensor_map.
        // The shard bytes are freed at the end of this block.
        {
            let file_bytes = std::fs::read(path)?;
            let tensors = safetensors::SafeTensors::deserialize(&file_bytes)?;
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
                    safetensors::Dtype::F8_E4M3 => data
                        .iter()
                        .map(|&b| crate::model::fp8_e4m3_to_f32(b))
                        .collect(),
                    _ => continue,
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
        let embed_key = "model.language_model.embed_tokens.weight";
        if !embed_loaded && tensor_map.contains_key(embed_key) {
            eprintln!("Loading embedding weights...");
            let embed_weight = take_tensor_2d(&mut tensor_map, embed_key, device)?;
            if tc.tie_word_embeddings {
                lm_head_weight = Some(embed_weight.clone().transpose());
            }
            transformer = transformer.load_embed_tokens(embed_weight);
            embed_loaded = true;
        }

        // Load any text layers whose tensors are now all available.
        while next_text_layer < tc.num_hidden_layers
            && vl_text_layer_ready(next_text_layer, &tensor_map)
        {
            if next_text_layer.is_multiple_of(10) {
                eprintln!(
                    "Loading text layer {}/{}...",
                    next_text_layer, tc.num_hidden_layers
                );
            }
            transformer =
                load_vl_text_layer(&mut tensor_map, transformer, next_text_layer, device)?;
            next_text_layer += 1;
        }

        // Load patch embedding as soon as it's available.
        let patch_key = "model.visual.patch_embed.proj.weight";
        let patch_bias_key = "model.visual.patch_embed.proj.bias";
        if tensor_map.contains_key(patch_key) && tensor_map.contains_key(patch_bias_key) {
            let (data, shape) = tensor_map.remove(patch_key).unwrap();
            let out_dim = shape[0];
            let in_dim: usize = shape[1..].iter().product();
            let weight = Tensor::<B, 1>::from_floats(&data[..], device).reshape([out_dim, in_dim]);
            let bias = take_tensor_1d(&mut tensor_map, patch_bias_key, device)?;
            vision_encoder = vision_encoder.load_patch_embed(weight, bias);
        }

        // Load position embeddings as soon as available.
        let pos_key = "model.visual.pos_embed.weight";
        if tensor_map.contains_key(pos_key) {
            let pos_weight = take_tensor_2d(&mut tensor_map, pos_key, device)?;
            vision_encoder = vision_encoder.load_pos_embed(pos_weight);
        }

        // Load any vision blocks whose tensors are now all available.
        while next_vision_block < config.vision_config.depth
            && vision_block_ready(next_vision_block, &tensor_map)
        {
            if next_vision_block.is_multiple_of(10) {
                eprintln!(
                    "Loading vision block {}/{}...",
                    next_vision_block, config.vision_config.depth
                );
            }
            let p = format!("model.visual.blocks.{}", next_vision_block);
            let norm1_w = take_tensor_1d(&mut tensor_map, &format!("{}.norm1.weight", p), device)?;
            let norm1_b = take_tensor_1d(&mut tensor_map, &format!("{}.norm1.bias", p), device)?;
            let norm2_w = take_tensor_1d(&mut tensor_map, &format!("{}.norm2.weight", p), device)?;
            let norm2_b = take_tensor_1d(&mut tensor_map, &format!("{}.norm2.bias", p), device)?;
            let qkv_w =
                take_linear_weight(&mut tensor_map, &format!("{}.attn.qkv.weight", p), device)?;
            let qkv_b = take_tensor_1d(&mut tensor_map, &format!("{}.attn.qkv.bias", p), device)?;
            let proj_w =
                take_linear_weight(&mut tensor_map, &format!("{}.attn.proj.weight", p), device)?;
            let proj_b = take_tensor_1d(&mut tensor_map, &format!("{}.attn.proj.bias", p), device)?;
            let fc1_w = take_linear_weight(
                &mut tensor_map,
                &format!("{}.mlp.linear_fc1.weight", p),
                device,
            )?;
            let fc1_b = take_tensor_1d(
                &mut tensor_map,
                &format!("{}.mlp.linear_fc1.bias", p),
                device,
            )?;
            let fc2_w = take_linear_weight(
                &mut tensor_map,
                &format!("{}.mlp.linear_fc2.weight", p),
                device,
            )?;
            let fc2_b = take_tensor_1d(
                &mut tensor_map,
                &format!("{}.mlp.linear_fc2.bias", p),
                device,
            )?;
            vision_encoder.load_block(
                next_vision_block,
                norm1_w,
                norm1_b,
                norm2_w,
                norm2_b,
                qkv_w,
                qkv_b,
                proj_w,
                proj_b,
                fc1_w,
                fc1_b,
                fc2_w,
                fc2_b,
            );
            next_vision_block += 1;
        }

        // Load main merger as soon as available.
        let merger_prefix = "model.visual.merger";
        if !merger_loaded && merger_ready(merger_prefix, &tensor_map) {
            let (norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b) =
                load_merger_weights(&mut tensor_map, merger_prefix, device)?;
            vision_encoder.load_merger(norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b);
            merger_loaded = true;
        }

        // Load any deepstack mergers whose tensors are now all available.
        while next_ds_merger < ds_indexes.len() {
            let dp = format!("model.visual.deepstack_merger_list.{}", next_ds_merger);
            if !merger_ready(&dp, &tensor_map) {
                break;
            }
            let (norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b) =
                load_merger_weights(&mut tensor_map, &dp, device)?;
            vision_encoder.load_deepstack_merger(
                next_ds_merger,
                norm_w,
                norm_b,
                fc1_w,
                fc1_b,
                fc2_w,
                fc2_b,
            );
            next_ds_merger += 1;
        }

        // Load final norm and lm_head as soon as available.
        let norm_key = "model.language_model.norm.weight";
        if tensor_map.contains_key(norm_key) {
            let norm_weight = take_tensor_1d(&mut tensor_map, norm_key, device)?;
            transformer = transformer.load_norm(norm_weight);
        }

        // Load lm_head if stored explicitly (some FP8 models store it even with
        // tie_word_embeddings=true since the head uses a different dtype).
        for lm_key in ["model.language_model.lm_head.weight", "lm_head.weight"] {
            if tensor_map.contains_key(lm_key) {
                let w = take_linear_weight(&mut tensor_map, lm_key, device)?;
                lm_head_weight = Some(w);
                break;
            }
        }
    }

    // Load tied lm_head weight if applicable.
    if let Some(w) = lm_head_weight {
        transformer = transformer.load_lm_head(w);
    }

    // Log any unloaded tensors for debugging
    if !tensor_map.is_empty() {
        eprintln!("Warning: {} tensors not loaded:", tensor_map.len());
        let mut keys: Vec<&String> = tensor_map.keys().collect();
        keys.sort();
        for k in keys.iter().take(20) {
            eprintln!("  - {}", k);
        }
        if keys.len() > 20 {
            eprintln!("  ... and {} more", keys.len() - 20);
        }
    }

    eprintln!("Model loaded successfully.");
    Ok((transformer, vision_encoder))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_vl_2b_preset() {
        let cfg = Qwen3VLConfig::qwen3_vl_2b();
        assert_eq!(cfg.text_config.hidden_size, 2048);
        assert_eq!(cfg.text_config.num_hidden_layers, 28);
        assert_eq!(cfg.text_config.head_dim, 64);
        assert_eq!(cfg.text_config.rope_theta, 5_000_000.0);
        assert_eq!(cfg.vision_config.hidden_size, 1024);
        assert_eq!(cfg.vision_config.depth, 24);
        assert_eq!(cfg.image_token_id, 151655);
        assert_eq!(cfg.video_token_id, 151656);
    }

    #[test]
    fn qwen3_vl_config_deserialization() {
        let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 8192,
                "vocab_size": 151936,
                "head_dim": 64,
                "rope_theta": 5000000.0,
                "tie_word_embeddings": true,
                "mrope_section": [24, 20, 20]
            },
            "vision_config": {
                "hidden_size": 1024,
                "out_hidden_size": 2048,
                "depth": 24,
                "num_heads": 16,
                "intermediate_size": 4096,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            },
            "image_token_id": 151655,
            "video_token_id": 151656
        }"#;
        let cfg: Qwen3VLConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.text_config.hidden_size, 2048);
        assert_eq!(cfg.text_config.mrope_section, Some(vec![24, 20, 20]));
        assert_eq!(cfg.vision_config.hidden_size, 1024);
        assert_eq!(cfg.image_token_id, 151655);
    }

    #[test]
    fn text_config_defaults() {
        let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "intermediate_size": 8192,
                "vocab_size": 151936
            },
            "vision_config": {}
        }"#;
        let cfg: Qwen3VLConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.text_config.rms_norm_eps, 1e-6);
        assert_eq!(cfg.text_config.head_dim, 64);
        assert_eq!(cfg.text_config.rope_theta, 5_000_000.0);
        assert!(!cfg.text_config.tie_word_embeddings);
        assert_eq!(cfg.text_config.mrope_section, None);
        // Default special tokens
        assert_eq!(cfg.image_token_id, 151655);
    }
}
