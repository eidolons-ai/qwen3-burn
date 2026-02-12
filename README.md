# qwen3-burn

Qwen3 inference in Rust, built on the [Burn](https://burn.dev) deep learning framework.

Loads weights from HuggingFace SafeTensors or GGUF (pre-quantized) files. Supports INT8/INT4 weight quantization. Runs on Metal (macOS), CUDA (NVIDIA), or CPU.

## Quick Start

**1. Install Rust** (if needed): https://rustup.rs

**2. Download a model:**

```bash
pip install huggingface-hub

# Option A: SafeTensors (full precision, ~1.5 GB)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', local_dir='./models/Qwen3-0.6B',
    allow_patterns=['*.safetensors', 'config.json', 'tokenizer.json'])
"

# Option B: GGUF (pre-quantized Q8_0, ~639 MB — also needs tokenizer.json from base model)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Qwen/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-Q8_0.gguf', local_dir='./models/Qwen3-0.6B-GGUF')
hf_hub_download('Qwen/Qwen3-0.6B', 'tokenizer.json', local_dir='./models/Qwen3-0.6B-GGUF')
"
```

**3. Run:**

```bash
# SafeTensors — macOS (Metal)
cargo run --release --features wgpu --example chat -- \
  --model-path ./models/Qwen3-0.6B \
  --prompt "Explain quicksort in one sentence"

# GGUF — auto-detected from .gguf extension
cargo run --release --features wgpu --example chat -- \
  --model-path ./models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf \
  --prompt "Explain quicksort in one sentence"

# CPU (slower, no GPU required)
cargo run --release --features ndarray --example chat -- \
  --model-path ./models/Qwen3-0.6B \
  --prompt "Hello"

# CUDA
cargo run --release --features cuda --example chat -- \
  --model-path ./models/Qwen3-0.6B \
  --prompt "Hello"
```

## Chat Example Options

```
--model-path PATH    Model directory or .gguf file (required)
--prompt TEXT        Input prompt (default: "What is the capital of France?")
--temperature FLOAT  Sampling temperature, 0.0 = greedy (default: 0.6)
--top-p FLOAT        Nucleus sampling threshold (default: 0.9)
-n, --max-tokens N   Max tokens to generate (default: 256)
--max-seq-len N      Context window size (default: 2048)
--chunk-size N       Prefill chunk size in tokens (default: full prompt at once)
--quantize MODE      Weight quantization: none, int8, int4 (default: none)
--format FORMAT      Model format: auto, safetensors, gguf (default: auto)
--seed N             RNG seed (default: 42)
```

## Library Usage

```rust
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::f16;
use qwen3_burn::model::Qwen3;
use qwen3_burn::QuantizationMode;
use qwen3_burn::sampling::Sampler;
use qwen3_burn::tokenizer::Qwen3Tokenizer;

type Backend = Wgpu<f16, i32>;  // f16 for 2x memory savings + faster Metal/Vulkan

let device = WgpuDevice::default();
let tokenizer = Qwen3Tokenizer::new("./models/Qwen3-0.6B/tokenizer.json").unwrap();
let mut model = Qwen3::<Backend>::from_pretrained(
    "./models/Qwen3-0.6B", 2048, QuantizationMode::None, &device,
).unwrap();
let mut sampler = Sampler::new_top_p(0.9, 42);

let prompt = tokenizer.apply_chat_template("You are a helpful assistant.", "What is 2+2?");
let output = model.generate(&tokenizer, &prompt, 256, 0.6, &mut sampler).unwrap();
println!("{}", output.text);
```

### Streaming Generation

Use `generate_streaming` for token-by-token output and optional chunked prefill:

```rust
use std::ops::ControlFlow;
use qwen3_burn::model::{GenerationEvent, GenerationParams};

let output = model.generate_streaming(
    &tokenizer,
    GenerationParams {
        prompt: &prompt,
        max_new_tokens: 256,
        temperature: 0.6,
        sampler: &mut sampler,
        prefill_chunk_size: Some(512), // or None for full-prompt prefill
    },
    |event| {
        if let GenerationEvent::Token { ref text, .. } = event {
            print!("{}", text);
        }
        ControlFlow::Continue(()) // return Break(()) to cancel early
    },
).unwrap();
```

## Quantization

Reduce memory usage with INT8/INT4 weight quantization (PackedU32 storage, requires GPU backend):

```rust
use qwen3_burn::QuantizationMode;

// INT8: ~4x memory reduction, minimal quality loss
let mut model = Qwen3::<Wgpu>::from_pretrained(
    "./models/Qwen3-8B", 2048, QuantizationMode::Int8, &device,
).unwrap();

// INT4: ~8x memory reduction
let mut model = Qwen3::<Wgpu>::from_pretrained(
    "./models/Qwen3-8B", 2048, QuantizationMode::Int4, &device,
).unwrap();
```

**GGUF auto-quantization**: When loading Q8_0 or Q4_0 GGUF files, quantization is auto-detected — weights are loaded directly into packed quantized format per-tensor, avoiding a full f32 model on GPU. No `--quantize` flag needed:

```bash
# Auto-quantized: ~3.2 GB RSS for Qwen3-0.6B Q8_0
cargo run --release --features wgpu --example chat -- \
  --model-path ./models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --prompt "Hello"

# SafeTensors with explicit quantization
cargo run --release --features wgpu --example chat -- \
  --model-path ./models/Qwen3-8B --prompt "Hello" --quantize int8
```

| Mode | Memory | Quality | Backend Support |
|------|--------|---------|-----------------|
| `none` | Full (FP32) | Best | All |
| `int8` | ~1/4 | Very good | WGPU, CUDA |
| `int4` | ~1/8 | Good | WGPU, CUDA |

## Vision (Qwen3-VL)

The `vision_chat` example supports Qwen3-VL vision-language models for image and video understanding. Requires the `vision` feature.

**Download a model:**

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-2B-Thinking-FP8', local_dir='./models/Qwen3-VL-2B-Thinking-FP8',
    allow_patterns=['*.safetensors', 'config.json', 'tokenizer.json'])
"
```

**Image input:**

```bash
cargo run --release --features "wgpu,vision" --example vision_chat -- \
  --model-path ./models/Qwen3-VL-2B-Thinking-FP8 \
  --image photo.jpg \
  --prompt "What do you see in this image?"
```

**Video input** (pre-extracted frames):

The `--video-frames` flag accepts individual image files, not video containers (.mov, .mp4). Extract frames with ffmpeg first:

```bash
# Extract ~8 evenly-spaced frames from a video
ffmpeg -i video.mov -vf "fps=1" -frames:v 8 frames/frame_%04d.png

cargo run --release --features "wgpu,vision" --example vision_chat -- \
  --model-path ./models/Qwen3-VL-2B-Thinking-FP8 \
  --video-frames frames/*.png \
  --prompt "What do you see in this video?" \
  --max-seq-len 16384
```

**Limitations:**
- No native video decoding — frames must be pre-extracted as PNG/JPEG images
- More frames = more tokens and longer prefill. 4-8 frames is a practical starting point; 17 frames at full resolution produced ~8700 tokens and 32s prefill on Apple Silicon
- `--max-seq-len` must be large enough to fit all vision tokens plus the text prompt (the default 4096 is often too small for video; use 8192-16384)
- Frames must have an even count (temporal patch size = 2); odd counts are padded automatically
- Only batch size 1 is supported

**Vision chat options:**

```
--image PATH         Image file(s), repeatable (PNG/JPEG)
--video-frames PATH  Video frame files (multiple, shell glob OK)
--max-seq-len N      Must accommodate vision tokens (default: 4096)
```

## Supported Models

Both dense and Mixture of Experts (MoE) Qwen3 models are supported. Preset configs are provided for:

| Model | Params | Active Params | Type | `from_pretrained` repo |
|-------|--------|---------------|------|----------------------|
| Qwen3-0.6B | 0.6B | 0.6B | Dense | `Qwen/Qwen3-0.6B` |
| Qwen3-1.7B | 1.7B | 1.7B | Dense | `Qwen/Qwen3-1.7B` |
| Qwen3-4B | 4B | 4B | Dense | `Qwen/Qwen3-4B` |
| Qwen3-8B | 8B | 8B | Dense | `Qwen/Qwen3-8B` |
| Qwen3-30B-A3B | 30B | 3B | MoE | `Qwen/Qwen3-30B-A3B` |
| Qwen3-235B-A22B | 235B | 22B | MoE | `Qwen/Qwen3-235B-A22B` |

MoE models use 128 experts with top-8 routing per token.

**SafeTensors**: model directory must contain `config.json`, `tokenizer.json`, and `*.safetensors`.

**GGUF**: a single `.gguf` file plus `tokenizer.json` in the same directory. Config is extracted from GGUF metadata. Supported quantization types: F32, F16, BF16, Q8_0, Q4_0.

## Testing

```bash
cargo test            # unit tests, no GPU or model weights needed
cargo fmt -- --check  # formatting
cargo clippy --all-targets  # lints (example warnings are expected without a backend feature)
```

## Benchmarks

Criterion benchmarks cover all core operations using Qwen3-0.6B dimensions on CPU (NdArray backend). No GPU or model weights needed.

```bash
cargo bench --features bench              # All benchmarks
cargo bench --features bench -- rms_norm  # Single group
cargo bench --features bench -- "attention/decode"  # Subset
```

Benchmark groups: `rms_norm`, `rope`, `feed_forward`, `moe_layer`, `attention`, `transformer_block`, `causal_mask`, `kv_cache`. Each group sweeps sequence lengths (1 to 512). Attention and transformer block groups include both prefill and decode scenarios. HTML reports are written to `target/criterion/`.

## Backends

| Feature | Backend | Precision | Notes |
|---------|---------|-----------|-------|
| `wgpu` | Metal / Vulkan / WebGPU | f16 | Best for macOS (Metal auto-selected) |
| `ndarray` | CPU | f32 | No GPU required, slower |
| `cuda` | NVIDIA CUDA | f16 | Requires CUDA toolkit |

## License

MIT OR Apache-2.0
