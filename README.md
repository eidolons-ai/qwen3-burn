# qwen3-burn

Qwen3 inference in Rust, built on the [Burn](https://burn.dev) deep learning framework.

Loads weights directly from HuggingFace SafeTensors files. Runs on Metal (macOS), CUDA (NVIDIA), or CPU.

## Quick Start

**1. Install Rust** (if needed): https://rustup.rs

**2. Download a model:**

```bash
pip install huggingface-hub

# Qwen3-0.6B (~1.5 GB)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', local_dir='./models/Qwen3-0.6B',
    allow_patterns=['*.safetensors', 'config.json', 'tokenizer.json'])
"
```

**3. Run:**

```bash
# macOS (Metal)
cargo run --release --features wgpu --example chat -- \
  --model-path ./models/Qwen3-0.6B \
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
--model-path PATH    Model directory (required)
--prompt TEXT        Input prompt (default: "What is the capital of France?")
--temperature FLOAT  Sampling temperature, 0.0 = greedy (default: 0.6)
--top-p FLOAT        Nucleus sampling threshold (default: 0.9)
-n, --max-tokens N   Max tokens to generate (default: 256)
--max-seq-len N      Context window size (default: 2048)
--seed N             RNG seed (default: 42)
```

## Library Usage

```rust
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use qwen3_burn::model::Qwen3;
use qwen3_burn::sampling::Sampler;
use qwen3_burn::tokenizer::Qwen3Tokenizer;

let device = WgpuDevice::default();
let tokenizer = Qwen3Tokenizer::new("./models/Qwen3-0.6B/tokenizer.json").unwrap();
let mut model = Qwen3::<Wgpu>::from_pretrained("./models/Qwen3-0.6B", 2048, &device).unwrap();
let mut sampler = Sampler::new_top_p(0.9, 42);

let prompt = tokenizer.apply_chat_template("You are a helpful assistant.", "What is 2+2?");
let output = model.generate(&tokenizer, &prompt, 256, 0.6, &mut sampler);
println!("{}", output.text);
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

MoE models use 128 experts with top-8 routing per token. The model directory must contain `config.json`, `tokenizer.json`, and `*.safetensors`.

## Backends

| Feature | Backend | Notes |
|---------|---------|-------|
| `wgpu` | Metal / Vulkan / WebGPU | Best for macOS (Metal auto-selected) |
| `ndarray` | CPU | No GPU required, slower |
| `cuda` | NVIDIA CUDA | Requires CUDA toolkit |

## License

MIT OR Apache-2.0
