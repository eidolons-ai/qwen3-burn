use std::io::{self, Write};
use std::ops::ControlFlow;
use std::time::Instant;

use clap::Parser;
use qwen3_burn::model::{GenerationEvent, GenerationParams, Qwen3};
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

    /// Prefill chunk size (tokens per chunk; omit for full-prompt prefill)
    #[arg(long)]
    chunk_size: Option<usize>,

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

    // Generate with streaming output
    let mut prev_text_len = 0;
    let chunk_size = args.chunk_size;
    let result = model.generate_streaming(
        &tokenizer,
        GenerationParams {
            prompt: &prompt,
            max_new_tokens: args.max_tokens,
            temperature: args.temperature,
            sampler: &mut sampler,
            prefill_chunk_size: chunk_size,
        },
        |event| {
            match event {
                GenerationEvent::PrefillProgress {
                    chunks_completed,
                    chunks_total,
                    ..
                } => {
                    if chunks_total > 1 {
                        eprint!("\rPrefill {}/{}...", chunks_completed, chunks_total);
                        if chunks_completed == chunks_total {
                            eprintln!();
                        }
                    }
                }
                GenerationEvent::Token { ref text, .. } => {
                    // Print only the newly added characters
                    let new_text = &text[prev_text_len..];
                    print!("{}", new_text);
                    io::stdout().flush().unwrap();
                    prev_text_len = text.len();
                }
                GenerationEvent::Done {
                    tokens_generated,
                    total_time_secs,
                    prefill_time_secs,
                    stop_reason,
                } => {
                    println!();
                    eprintln!("---");
                    let decode_time = total_time_secs - prefill_time_secs;
                    let tps = if decode_time > 0.0 {
                        tokens_generated as f64 / decode_time
                    } else {
                        0.0
                    };
                    eprintln!(
                        "{} tokens generated ({:.2} tokens/s, prefill {:.2}s, stop: {:?})",
                        tokens_generated, tps, prefill_time_secs, stop_reason
                    );
                }
            }
            ControlFlow::Continue(())
        },
    );
    match result {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
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
