use std::io::{self, Write};
use std::time::Instant;

use clap::Parser;
use qwen3_burn::model::Qwen3;
use qwen3_burn::sampling::Sampler;
use qwen3_burn::tokenizer::Qwen3Tokenizer;

const DEFAULT_PROMPT: &str = "What is the capital of France?";
const SYSTEM_PROMPT: &str = "You are a helpful assistant.";

#[derive(Parser, Debug)]
#[command(name = "qwen3-chat", about = "Qwen3 chat example using Burn")]
struct Args {
    /// Path to the model directory (containing config.json, tokenizer.json, *.safetensors)
    #[arg(short, long)]
    model_path: String,

    /// Input prompt
    #[arg(short, long, default_value = DEFAULT_PROMPT)]
    prompt: String,

    /// Temperature for sampling (0.0 = greedy)
    #[arg(short, long, default_value_t = 0.6)]
    temperature: f64,

    /// Top-p probability threshold
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Maximum new tokens to generate
    #[arg(short = 'n', long, default_value_t = 256)]
    max_tokens: usize,

    /// Maximum sequence length (prompt + generation)
    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn run<B: burn::prelude::Backend>(args: Args, device: burn::prelude::Device<B>) {
    eprintln!("Loading tokenizer...");
    let tokenizer_path = std::path::Path::new(&args.model_path).join("tokenizer.json");
    let tokenizer = Qwen3Tokenizer::new(&tokenizer_path).expect("Failed to load tokenizer");

    eprintln!("Loading model from {}...", args.model_path);
    let load_start = Instant::now();
    let mut model = Qwen3::<B>::from_pretrained(&args.model_path, args.max_seq_len, &device)
        .expect("Failed to load model");
    eprintln!("Model loaded in {:.1}s", load_start.elapsed().as_secs_f64());

    // Format prompt with chat template
    let prompt = tokenizer.apply_chat_template(SYSTEM_PROMPT, &args.prompt);
    eprintln!("Prompt: {}", args.prompt);
    eprintln!("---");

    // Set up sampler
    let mut sampler = if args.temperature > 0.0 {
        Sampler::new_top_p(args.top_p, args.seed)
    } else {
        Sampler::Argmax
    };

    // Generate
    let result = model.generate(
        &tokenizer,
        &prompt,
        args.max_tokens,
        args.temperature,
        &mut sampler,
    );

    // Print result
    print!("{}", result.text);
    io::stdout().flush().unwrap();
    println!();

    eprintln!("---");
    eprintln!(
        "{} tokens generated ({:.2} tokens/s)",
        result.tokens,
        result.tokens as f64 / result.time
    );
}

fn main() {
    let args = Args::parse();

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::wgpu::WgpuDevice;
        use burn::backend::Wgpu;
        let device = WgpuDevice::default();
        run::<Wgpu>(args, device);
    }

    #[cfg(feature = "ndarray")]
    {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::backend::NdArray;
        let device = NdArrayDevice::Cpu;
        run::<NdArray>(args, device);
    }

    #[cfg(feature = "cuda")]
    {
        use burn::backend::cuda::CudaDevice;
        use burn::backend::Cuda;
        use burn::tensor::f16;
        let device = CudaDevice::default();
        run::<Cuda<f16, i32>>(args, device);
    }

    #[cfg(not(any(feature = "wgpu", feature = "ndarray", feature = "cuda")))]
    {
        eprintln!("No backend feature enabled. Use --features wgpu, ndarray, or cuda.");
        std::process::exit(1);
    }
}
