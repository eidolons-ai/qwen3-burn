# CLAUDE.md

## Project Overview

qwen3-burn is a Rust library implementing Qwen3 LLM inference using the Burn 0.20 deep learning framework. It loads HuggingFace SafeTensors weights directly and runs on multiple backends (WGPU/Metal, NdArray/CPU, CUDA).

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
  lib.rs           # Re-exports: model, sampling, tokenizer (cache & transformer are pub(crate))
  model.rs         # Qwen3Config (serde from config.json), Qwen3<B> (generation), SafeTensors loading
  transformer.rs   # Transformer, TransformerBlock, Mlp (Dense/Moe), MoeLayer, MultiHeadAttention, FeedForward, RmsNorm, RotaryEmbedding
  cache.rs         # KvCache - pre-allocated [batch, heads, max_seq, head_dim] with sliding window
  sampling.rs      # Sampler enum: TopP (nucleus) and Argmax
  tokenizer.rs     # Qwen3Tokenizer wrapping HF `tokenizers` crate
examples/
  chat.rs          # CLI chat app with clap, feature-gated backend selection
```

## Weight Loading

PyTorch stores Linear weights as `[out_features, in_features]`; Burn stores as `[in_features, out_features]`. All Linear weights are **transposed** during loading (`get_linear_weight`). Embedding weights are loaded without transpose (`get_tensor_2d_raw`). When `tie_word_embeddings=true`, the embedding weight is transposed and used as the lm_head weight.

SafeTensors key mapping (dense layers):
- `model.embed_tokens.weight` -> Embedding
- `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` -> Linear (transposed)
- `model.layers.{i}.self_attn.{q,k}_norm.weight` -> RmsNorm (1D)
- `model.layers.{i}.mlp.{gate,up,down}_proj.weight` -> Linear (transposed)
- `model.layers.{i}.{input,post_attention}_layernorm.weight` -> RmsNorm (1D)
- `model.norm.weight` -> final RmsNorm
- `lm_head.weight` -> Linear (transposed, or tied with embedding)

SafeTensors key mapping (MoE layers):
- `model.layers.{i}.mlp.experts.gate_up_proj` -> 3D `[num_experts, 2*moe_intermediate, hidden]` (per-expert transposed via swap_dims)
- `model.layers.{i}.mlp.experts.down_proj` -> 3D `[num_experts, hidden, moe_intermediate]` (per-expert transposed via swap_dims)
- `model.layers.{i}.mlp.router.weight` -> 2D `[num_experts, hidden]` (no transpose)

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

# Run (needs model files in a directory: config.json, tokenizer.json, *.safetensors)
cargo run --release --features wgpu --example chat -- \
  --model-path ./models/Qwen3-0.6B --prompt "Hello" --max-tokens 100

# Download model files (Python)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='./models/Qwen3-0.6B', allow_patterns=['*.safetensors', 'config.json', 'tokenizer.json'])"
```

Backend features are mutually exclusive: `wgpu` (Metal on macOS), `ndarray` (CPU), `cuda`.

Tests use `NdArray` backend (CPU) via dev-dependencies and require no model weights. 47 unit tests cover config parsing, RmsNorm, RoPE, causal mask, KV cache, MoE routing, sampling, and end-to-end shape verification for both dense and MoE transformer blocks.

## Key Implementation Details

- RmsNorm operates on 3D tensors `[batch, seq, hidden]`, computing variance over dim 2
- QK-Norm reshapes Q/K to `[batch*seq*heads, 1, head_dim]` to apply RmsNorm on head_dim, then reshapes back
- RoPE precomputes cos/sin tables at init; slices per-position during forward
- Causal mask is built on CPU as a float tensor with 0.0 / -inf values, shaped `[q_seq, kv_seq]`, then unsqueezed to `[1, 1, q_seq, kv_seq]` for broadcasting
- KV cache uses pre-allocated tensors with `slice_assign` for appending; sliding window shifts when exceeding max_seq_len
- Generation loop: full prompt in one forward pass, then one token at a time with cache

### MoE Implementation

- `Mlp` enum (`Dense`/`Moe`) makes TransformerBlock polymorphic — each layer is independently dense or MoE
- `MoeLayer` stores packed 3D expert weights (gate_up_proj, down_proj) and a 2D router weight
- Forward: router softmax → top-k selection → optional renormalization (`norm_topk_prob`) → per-expert gather/SwiGLU/scatter
- Expert routing iterates over experts on CPU (index selection), dispatches GPU matmuls per active expert
- `Qwen3Config::is_moe_layer(i)` determines per-layer type via `decoder_sparse_step` and `mlp_only_layers`

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

Qwen3-0.6B on Apple Silicon (M-series, Metal via WGPU): ~15-18 tokens/s, model loads in <1s.

## Qwen3 Chat Template

```
<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n
```

Special tokens: BOS=151643, EOS=151645. The 0.6B model generates `<think>...</think>` blocks before answering (thinking mode).
