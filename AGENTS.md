# CLAUDE.md

## Project Overview

qwen3-burn is a Rust library implementing Qwen3 LLM inference using the Burn 0.20 deep learning framework. It loads HuggingFace SafeTensors or GGUF pre-quantized weights and runs on multiple backends (WGPU/Metal, NdArray/CPU, CUDA). The WGPU and CUDA backends use f16 inference for 2x memory reduction and higher throughput on supported hardware.

## Architecture

Qwen3 is a decoder-only transformer with these distinguishing features vs Llama:
- **QK-Norm**: RMSNorm applied to Q and K projections *before* RoPE (unique to Qwen3)
- **GQA**: Grouped-query attention with consistently 8 KV heads (dense) or 4 KV heads (MoE) across model sizes
- **SwiGLU**: `down_proj(silu(gate_proj(x)) * up_proj(x))`
- **RoPE**: theta=1,000,000, head_dim=128
- **RMSNorm**: eps=1e-6 (Llama uses 1e-5)
- **No bias** in any linear layers
- **vocab_size**: 151,936 (dense) or 152,064 (235B MoE)
- **tie_word_embeddings**: true for <=4B and 30B-A3B, false for >=8B and 235B-A22B
- **MoE** (30B-A3B, 235B-A22B): Router + 128 expert FFNs per layer, top-8 routing per token

## Module Layout

```
src/
  lib.rs           # Re-exports: model, sampling, tokenizer, QuantizationMode (cache, gguf & transformer are pub(crate)); bench_internals feature gate
  model.rs         # Qwen3Config, Qwen3<B> (generate/generate_streaming/from_gguf), StopReason, GenerationEvent, GenerationParams, QuantizationMode, SafeTensors & GGUF loading
  gguf.rs          # GGUF binary format parser, dequantization (F32/F16/BF16/Q8_0/Q4_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K), config extraction, tensor name mapping
  transformer.rs   # Transformer, TransformerBlock, Mlp (Dense/Moe), MoeLayer, MultiHeadAttention, FeedForward, RmsNorm, RotaryEmbedding
  cache.rs         # KvCache - pre-allocated [batch, heads, max_seq, head_dim] with sliding window
  sampling.rs      # Sampler enum: TopP (nucleus) and Argmax
  tokenizer.rs     # Qwen3Tokenizer wrapping HF `tokenizers` crate
examples/
  chat.rs          # CLI chat app with clap, feature-gated backend selection, --format auto/safetensors/gguf
benches/
  ops.rs           # Criterion benchmarks for all core ops (NdArray backend)
```

## Weight Loading

PyTorch stores Linear weights as `[out_features, in_features]`; Burn stores as `[in_features, out_features]`. All Linear weights are **transposed** during loading (`take_linear_weight`). Embedding weights are loaded without transpose (`take_tensor_2d`). When `tie_word_embeddings=true`, the embedding weight is transposed and used as the lm_head weight.

Weight loading streams shard-by-shard: after reading each safetensors file, completed layers are loaded into the model immediately and their f32 data is freed via `remove()` from the tensor map. This keeps peak memory close to the final model size (~1x) rather than ~2-3x.

### SafeTensors Format

SafeTensors key mapping (dense layers):
- `model.embed_tokens.weight` -> Embedding
- `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` -> Linear (transposed)
- `model.layers.{i}.self_attn.{q,k}_norm.weight` -> RmsNorm (1D)
- `model.layers.{i}.mlp.{gate,up,down}_proj.weight` -> Linear (transposed)
- `model.layers.{i}.{input,post_attention}_layernorm.weight` -> RmsNorm (1D)
- `model.norm.weight` -> final RmsNorm
- `lm_head.weight` -> Linear (transposed, or tied with embedding)

SafeTensors key mapping (MoE layers):
- `model.layers.{i}.mlp.experts.{j}.gate_proj.weight` -> 2D `[moe_intermediate, hidden]` (per-expert, transposed and packed with up_proj into 3D gate_up_proj)
- `model.layers.{i}.mlp.experts.{j}.up_proj.weight` -> 2D `[moe_intermediate, hidden]` (per-expert, transposed and packed with gate_proj)
- `model.layers.{i}.mlp.experts.{j}.down_proj.weight` -> 2D `[hidden, moe_intermediate]` (per-expert, transposed and stacked into 3D)
- `model.layers.{i}.mlp.gate.weight` -> 2D `[num_experts, hidden]` (router, no transpose)

### GGUF Format

GGUF loading (`Qwen3::from_gguf()`) uses a custom zero-dependency parser in `src/gguf.rs`. Config is extracted from GGUF metadata (no `config.json` needed); only a `tokenizer.json` alongside the `.gguf` file is required.

**Critical**: GGUF stores tensor dimensions in **reversed order** (innermost-first, ggml convention). All dims are reversed when constructing Burn tensors.

Supported GGUF quantization types: F32, F16, BF16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.

When the GGUF source is quantized (Q8_0, Q4_0, or any k-quant type), `from_gguf()` auto-detects the quantization and loads 2D linear weights directly into PackedU32 quantized format via per-tensor CPU quantization (dequant → transpose → requant per tensor). This avoids holding a full f32 model on GPU. K-quant types Q5_K/Q6_K map to Int8; Q2_K/Q3_K/Q4_K map to Int4. The auto-detected mode can be overridden by the user's `QuantizationMode` argument; `QuantizationMode::None` upgrades to the detected mode. 1D weights (norms), embeddings, and 3D MoE expert weights always load as f32.

The architecture prefix is read from `general.architecture` in GGUF metadata (typically `qwen3` for official Qwen3 GGUFs, or `qwen2` for older conversions). Key metadata fields (shown with `{arch}` prefix):
- `{arch}.embedding_length` -> hidden_size
- `{arch}.block_count` -> num_hidden_layers
- `{arch}.attention.head_count` -> num_attention_heads
- `{arch}.attention.head_count_kv` -> num_key_value_heads
- `{arch}.feed_forward_length` -> intermediate_size
- `tie_word_embeddings` inferred from absence of `output.weight` tensor
- `vocab_size` inferred from `token_embd.weight` tensor shape

GGUF tensor name mapping: `token_embd.weight`, `output_norm.weight`, `output.weight`, `blk.{i}.attn_q.weight`, etc. MoE experts stored as separate `ffn_gate_exps`/`ffn_up_exps` tensors, fused into `gate_up_proj` during loading.

## Build & Test

```bash
# Check (fast)
cargo check --features wgpu

# Build release
cargo build --release --features wgpu --example chat

# Run tests (no GPU or model weights needed — uses ndarray backend via dev-dependencies)
cargo test

# Lint
cargo fmt -- --check
cargo clippy --all-targets

# Benchmarks (CPU, no model weights needed — uses ndarray backend + "bench" feature)
cargo bench --features bench              # All benchmarks
cargo bench --features bench -- rms_norm  # Single group
cargo bench --features bench -- "attention/decode"  # Subset

# Run with SafeTensors (needs config.json, tokenizer.json, *.safetensors)
cargo run --release --features wgpu --example chat -- \
  --model-path ./models/Qwen3-0.6B --prompt "Hello" --max-tokens 100

# Run with GGUF (auto-detected from .gguf extension; needs tokenizer.json next to .gguf file)
cargo run --release --features wgpu --example chat -- \
  --model-path ./models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --prompt "Hello" --max-tokens 100

# Run with INT8 quantization (~4x memory reduction)
cargo run --release --features wgpu --example chat -- \
  --model-path ./models/Qwen3-0.6B --prompt "Hello" --quantize int8

# Download SafeTensors model files (Python)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='./models/Qwen3-0.6B', allow_patterns=['*.safetensors', 'config.json', 'tokenizer.json'])"

# Download GGUF model (needs tokenizer.json from base model)
# huggingface-cli download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir ./models/Qwen3-0.6B-GGUF
# huggingface-cli download Qwen/Qwen3-0.6B tokenizer.json --local-dir ./models/Qwen3-0.6B-GGUF
```

Backend features are mutually exclusive: `wgpu` (Metal on macOS, f16), `ndarray` (CPU, f32), `cuda` (NVIDIA, f16). The `bench` feature is independent and only enables `bench_internals` re-exports for the benchmark harness.

## Commit Style

Always run `cargo fmt` before committing. Do not include model or agent attribution in commit messages (no "Co-Authored-By", "Generated by", etc.).

Tests use `NdArray` backend (CPU) via dev-dependencies and require no model weights. Unit tests cover config parsing, streaming types, RmsNorm, RoPE, causal mask, KV cache (including overflow/truncation edge cases), MoE routing, sampling, quantization, CPU quantization helpers (Q8S/Q4S block quantization), GGUF quantization auto-detection, GGUF dequantization (F32/F16/BF16/Q8_0/Q4_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K), GGUF tensor name mapping, GGUF metadata accessors, and end-to-end shape verification for both dense and MoE transformer blocks.

Criterion benchmarks (`benches/ops.rs`) measure all core ops parameterized by sequence length using Qwen3-0.6B dimensions. 48 benchmarks across 8 groups: rms_norm, rope, feed_forward, moe_layer, attention (prefill + decode), transformer_block (prefill + decode), causal_mask, kv_cache. Stateful ops (attention, transformer_block, kv_cache) use `iter_batched` with fresh caches per iteration. HTML reports are written to `target/criterion/`.

## Key Implementation Details

- RmsNorm operates on 3D tensors `[batch, seq, hidden]`, computing variance over dim 2. Pre-scales input by `max(|x|, dim=-1)` (clamped to >= 1.0) before squaring to prevent f16 overflow; the scale factor cancels mathematically: `(x/s) / rms(x/s) = x / rms(x)`
- QK-Norm reshapes Q/K to `[batch*seq*heads, 1, head_dim]` to apply RmsNorm on head_dim, then reshapes back
- RoPE precomputes cos/sin tables at init in f64 on CPU (Qwen3's theta=1e6 yields frequencies down to ~1.6e-6, below f16's normal range of ~6e-5); the final cos/sin values are in [-1, 1] and convert to f16 without loss. Tables are sliced per-position during forward
- Causal mask is built on CPU as a float tensor with 0.0 / -inf values, shaped `[q_seq, kv_seq]`, then unsqueezed to `[1, 1, q_seq, kv_seq]` for broadcasting
- KV cache uses pre-allocated tensors with `slice_assign` for appending; sliding window shifts when exceeding max_seq_len; chunks larger than max_seq_len are truncated to the last max_seq_len tokens
- Generation: `generate_streaming()` is the core method; `generate()` is a convenience wrapper. Both return `Result<GenerationOutput, String>`.
  - Only batch size 1 is supported
  - Validates prompt length against `max_seq_len`; returns `Err` if the prompt (after tokenization) exceeds the model's sequence limit
  - `max_new_tokens=0` returns immediately after prefill with no tokens generated
  - `temperature <= 0` forces argmax regardless of the configured sampler
  - Callback-based streaming via `FnMut(GenerationEvent) -> ControlFlow<()>` — callers receive `PrefillProgress`, `Token`, and `Done` events
  - Optional chunked prefill: splits prompt into chunks of `prefill_chunk_size` tokens, reducing peak attention memory from O(N^2) to O(C*N)
  - Prefill chunks use `build_causal_mask(chunk_len, total_seq_len)` with the correct positional offset; KV cache accumulates across chunks. Decode steps (seq_len=1) skip the mask entirely (pass `None`) since a single query can attend to all cached positions.
  - `ControlFlow::Break(())` from the callback cancels generation early (yields `StopReason::Cancelled`)

### Quantization

- `QuantizationMode` enum: `None` (default), `Int8`, `Int4`
- Uses Burn's native `Quantizer` with `QuantScheme` for real packed quantized storage (`PackedU32`)
- `SelectiveQuantizer` wraps `burn::module::Quantizer` in a `ModuleMapper<B>` that skips sensitive parameters: 1D tensors (RMSNorm weights) and embedding weights (`embed_tokens`)
- INT8: Q8S symmetric per-block (block size 32), `PackedU32` storage — requires GPU backend (WGPU/CUDA)
- INT4: Q4S symmetric per-block (block size 32), `PackedU32` storage — requires GPU backend (WGPU/CUDA)
- `from_pretrained()` accepts `QuantizationMode` and applies whole-model quantization via `SelectiveQuantizer` after loading
- `from_gguf()` auto-detects quantized types (Q8_0, Q4_0, and k-quants Q2_K–Q6_K) and applies per-tensor quantized loading (dequant → CPU transpose → CPU requant → GPU upload per tensor), skipping the whole-model `apply_quantization` pass. K-quant types Q5_K/Q6_K map to Int8; Q2_K/Q3_K/Q4_K map to Int4. Helper functions: `detect_gguf_quantization`, `quantize_q8s`/`quantize_q4s` (CPU block quantization), `make_2d_quantized` (transpose + quantize + `TensorData::quantized`)
- **Contiguous workaround** (SafeTensors path only): Weight tensors are loaded as `[out, in]` then `.transpose()`d to `[in, out]`, creating non-contiguous memory views. Burn's block quantization computes scales over physical memory blocks but dequantization applies them over logical blocks, causing ~500x error for non-contiguous tensors. `SelectiveQuantizer` forces contiguity via `to_data()`/`from_data()` round-trip before quantizing (upstream Burn bug). The GGUF path avoids this by transposing f32 data on CPU before quantizing.
- Note: WGPU has an open upstream issue (tracel-ai/burn#3997) that may affect quantization with autotune

### MoE Implementation

- `Mlp` enum (`Dense`/`Moe`) makes TransformerBlock polymorphic — each layer is independently dense or MoE
- `MoeLayer` stores packed 3D expert weights (gate_up_proj, down_proj) and a 2D router weight
- During loading, per-expert 2D weights from safetensors are transposed and packed into 3D tensors: gate_proj + up_proj → `gate_up_proj` [num_experts, hidden, 2*moe_intermediate], down_proj → `down_proj` [num_experts, moe_intermediate, hidden]
- Forward: router softmax → top-k selection → optional renormalization (`norm_topk_prob`) → two-path dispatch based on token count
- **Batched decode path** (`forward_batched`): for `num_tokens <= num_experts_per_tok` (decode is always 1 token). Uses GPU-side `select(0, ...)` to gather top-k expert weights, then runs a single batched matmul across all k experts simultaneously. ~6 GPU dispatches per layer instead of ~40. No CPU roundtrip for expert indices.
- **Per-expert prefill path** (`forward_per_expert`): for prefill (many tokens). Builds a dispatch table via `HashMap<expert_idx, Vec<(token_idx, weight)>>` in one pass, then iterates only active experts (typically 8-30 unique for a short prefill) instead of all 128.
- `Qwen3Config::is_moe_layer(i)` determines per-layer type via `decoder_sparse_step` and `mlp_only_layers`

### f16 Numerical Handling

Burn's backend type parameter (`Wgpu<f16, i32>`) sets the float precision for all tensors. Three operations require special handling to avoid f16 overflow/underflow:

1. **RoPE frequency table** (`transformer.rs`): theta=1e6 produces inverse frequencies down to ~1.6e-6, below f16 normal range (~6e-5). Computed in f64 on CPU at init; cos/sin values (in [-1, 1]) load to GPU losslessly.
2. **RmsNorm squaring** (`transformer.rs`): Hidden state values > 255 cause x² > 65504 (f16 max), overflowing to inf. Pre-scaling by `max(|x|)` keeps x² ≤ 1.0; the scale cancels in the normalization.
3. **Sampling softmax** (`model.rs`, `sampling.rs`): f16 softmax over ~152K vocab overflows the denominator. Logits are extracted to CPU f64 for temperature scaling, softmax, and sampling. `Sampler::sample_probs(&[f64])` provides the CPU-side top-p path.

All other operations (matmuls, attention, SwiGLU, embedding lookup) run natively in f16 on the GPU.

## Model Sizes

### Dense Models

| Model | Layers | Hidden | Heads | KV Heads | Intermediate |
|-------|--------|--------|-------|----------|--------------|
| 0.6B  | 28     | 1024   | 16    | 8        | 3072         |
| 1.7B  | 28     | 1536   | 16    | 8        | 4608         |
| 4B    | 36     | 2560   | 32    | 8        | 9728         |
| 8B    | 36     | 4096   | 32    | 8        | 12288        |

### MoE Models

| Model | Layers | Hidden | Heads | KV Heads | Experts | Top-K | MoE Intermediate |
|-------|--------|--------|-------|----------|---------|-------|------------------|
| 30B-A3B | 48   | 2048   | 32    | 4        | 128     | 8     | 768              |
| 235B-A22B | 94 | 4096   | 64    | 4        | 128     | 8     | 1536             |

## Performance

Qwen3-0.6B on Apple Silicon (M-series, Metal via WGPU, f16): ~23-26 tokens/s, model loads in <1s.

## Qwen3 Chat Template

```
<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n
```

Special tokens: BOS=151643, EOS=151645. The 0.6B model generates `<think>...</think>` blocks before answering (thinking mode).
