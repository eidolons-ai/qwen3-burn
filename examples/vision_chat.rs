use std::io::{self, Write};
use std::ops::ControlFlow;
use std::time::Instant;

use clap::Parser;
use qwen3_burn::image::ImageProcessor;
use qwen3_burn::model::GenerationEvent;
use qwen3_burn::sampling::Sampler;
use qwen3_burn::tokenizer::Qwen3Tokenizer;
use qwen3_burn::vision_model::{Qwen3VL, VLGenerationParams, VisionInput};

const DEFAULT_PROMPT: &str = "Describe this image in detail.";

#[derive(Parser, Debug)]
#[command(
    name = "qwen3-vl-chat",
    about = "Qwen3-VL vision-language chat example"
)]
struct Args {
    /// Path to the model directory (SafeTensors)
    #[arg(short, long)]
    model_path: String,

    /// Image file path(s) (repeatable)
    #[arg(short, long)]
    image: Vec<String>,

    /// Video frame file paths (multiple frames for video)
    #[arg(long)]
    video_frames: Vec<String>,

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

    /// Maximum sequence length
    #[arg(long, default_value_t = 4096)]
    max_seq_len: usize,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn run<B: burn::prelude::Backend>(args: Args, device: burn::prelude::Device<B>) {
    let model_path = std::path::Path::new(&args.model_path);

    // Load tokenizer
    let tokenizer_path = model_path.join("tokenizer.json");
    eprintln!("Loading tokenizer...");
    let tokenizer = Qwen3Tokenizer::new(&tokenizer_path).expect("Failed to load tokenizer");

    // Load model
    eprintln!("Loading Qwen3-VL model from {}...", args.model_path);
    let load_start = Instant::now();
    let mut model = Qwen3VL::<B>::from_pretrained(&args.model_path, args.max_seq_len, &device)
        .expect("Failed to load model");
    eprintln!("Model loaded in {:.1}s", load_start.elapsed().as_secs_f64());

    // Preprocess images
    let processor = ImageProcessor::default();
    let mut image_inputs: Vec<VisionInput> = Vec::new();

    for img_path in &args.image {
        eprintln!("Processing image: {}", img_path);
        let input = processor
            .preprocess_image(std::path::Path::new(img_path))
            .expect("Failed to preprocess image");
        eprintln!(
            "  Grid: {}x{}x{}, merge tokens: {}",
            input.grid_thw.0, input.grid_thw.1, input.grid_thw.2, input.num_merge_tokens
        );
        image_inputs.push(VisionInput {
            pixel_patches: input.pixel_patches,
            grid_thw: input.grid_thw,
            num_merge_tokens: input.num_merge_tokens,
            num_patches: input.num_patches,
            patch_embed_dim: input.patch_embed_dim,
        });
    }

    if !args.video_frames.is_empty() {
        eprintln!("Processing {} video frames...", args.video_frames.len());
        let frame_paths: Vec<&std::path::Path> = args
            .video_frames
            .iter()
            .map(|p| std::path::Path::new(p.as_str()))
            .collect();
        let input = processor
            .preprocess_video_frames(&frame_paths)
            .expect("Failed to preprocess video frames");
        eprintln!(
            "  Grid: {}x{}x{}, merge tokens: {}",
            input.grid_thw.0, input.grid_thw.1, input.grid_thw.2, input.num_merge_tokens
        );
        image_inputs.push(VisionInput {
            pixel_patches: input.pixel_patches,
            grid_thw: input.grid_thw,
            num_merge_tokens: input.num_merge_tokens,
            num_patches: input.num_patches,
            patch_embed_dim: input.patch_embed_dim,
        });
    }

    // Build prompt with vision placeholders
    // Format: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    //         <|im_start|>user\n<|vision_start|><|image_pad|>...<|vision_end|>\n{prompt}<|im_end|>\n
    //         <|im_start|>assistant\n
    let mut prompt_text = String::from(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
    );
    for img in &image_inputs {
        prompt_text.push_str("<|vision_start|>");
        for _ in 0..img.num_merge_tokens {
            prompt_text.push_str("<|image_pad|>");
        }
        prompt_text.push_str("<|vision_end|>");
    }
    prompt_text.push('\n');
    prompt_text.push_str(&args.prompt);
    prompt_text.push_str("<|im_end|>\n<|im_start|>assistant\n");

    let token_ids = tokenizer.encode(&prompt_text);
    eprintln!("Prompt tokens: {}", token_ids.len());
    eprintln!("---");

    // Set up sampler
    let mut sampler = if args.temperature > 0.0 {
        Sampler::new_top_p(args.top_p, args.seed)
    } else {
        Sampler::Argmax
    };

    // Generate
    let mut prev_text_len = 0;
    let result = model.generate_with_vision(
        &tokenizer,
        &token_ids,
        &image_inputs,
        VLGenerationParams {
            prompt: &prompt_text,
            max_new_tokens: args.max_tokens,
            temperature: args.temperature,
            sampler: &mut sampler,
            prefill_chunk_size: None,
        },
        |event| {
            match event {
                GenerationEvent::PrefillProgress { .. } => {}
                GenerationEvent::Token { ref text, .. } => {
                    let stable_end = text.trim_end_matches('\u{FFFD}').len();
                    if stable_end > prev_text_len {
                        let start = text.floor_char_boundary(prev_text_len);
                        print!("{}", &text[start..stable_end]);
                        io::stdout().flush().unwrap();
                        prev_text_len = stable_end;
                    }
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
        use burn::tensor::f16;
        let device = WgpuDevice::default();
        run::<Wgpu<f16, i32>>(args, device);
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
