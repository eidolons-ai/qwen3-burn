use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;

use qwen3_burn::model::{QuantizationMode, Qwen3};
use qwen3_burn::sampling::Sampler;
use qwen3_burn::tokenizer::Qwen3Tokenizer;

type B = NdArray;

fn model_dir() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.join("models").join("Qwen3-0.6B-GGUF")
}

fn gguf_path() -> std::path::PathBuf {
    model_dir().join("Qwen3-0.6B-Q8_0.gguf")
}

fn tokenizer_path() -> std::path::PathBuf {
    model_dir().join("tokenizer.json")
}

#[test]
#[ignore]
fn smoke_gguf_generate() {
    let gguf = gguf_path();
    let tok = tokenizer_path();
    if !gguf.exists() || !tok.exists() {
        eprintln!(
            "Skipping smoke test: model files not found at {}",
            model_dir().display()
        );
        return;
    }

    let device = NdArrayDevice::Cpu;
    let tokenizer = Qwen3Tokenizer::new(&tok).expect("Failed to load tokenizer");

    eprintln!("Loading model (f32, no quantization)...");
    let mut model = Qwen3::<B>::from_gguf(&gguf, 2048, QuantizationMode::None, &device)
        .expect("Failed to load GGUF model");

    let prompt = tokenizer.apply_chat_template(
        "You are a helpful assistant.",
        "What is the capital of France?",
    );

    let mut sampler = Sampler::Argmax;
    let result = model
        .generate(&tokenizer, &prompt, 32, 0.0, &mut sampler)
        .expect("Generation failed");

    eprintln!("Generated text: {:?}", result.text);
    eprintln!("Tokens: {}, Time: {:.2}s", result.tokens, result.time);

    assert!(
        !result.text.is_empty(),
        "Generated text should not be empty"
    );
    assert!(result.tokens > 0, "Should generate at least one token");

    // Deterministic argmax output for regression detection.
    // The model produces a <think> block first, so we check for a known prefix.
    assert!(
        result.text.starts_with("<think>"),
        "Expected output to start with '<think>', got: {:?}",
        &result.text[..result.text.len().min(80)]
    );
}
