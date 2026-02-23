use std::path::Path;

/// Convenience wrapper around HuggingFace tokenizers for Qwen3.
///
/// This is a thin codec wrapper. The model owns special token semantics (BOS/EOS);
/// consumers who already have a `tokenizers::Tokenizer` can pass it directly to
/// [`Qwen3::generate`] / [`Qwen3::generate_streaming`] without constructing this type.
pub struct Qwen3Tokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Qwen3Tokenizer {
    /// Load the tokenizer from a `tokenizer.json` file.
    pub fn new(tokenizer_path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(Self { tokenizer })
    }

    /// Return a reference to the inner `tokenizers::Tokenizer`.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .expect("Failed to encode text");
        encoding.get_ids().to_vec()
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer
            .decode(tokens, true)
            .expect("Failed to decode tokens")
    }

    /// Format a user message using Qwen3's chat template.
    pub fn apply_chat_template(&self, system_prompt: &str, user_message: &str) -> String {
        format!(
            "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        )
    }
}

impl std::ops::Deref for Qwen3Tokenizer {
    type Target = tokenizers::Tokenizer;

    fn deref(&self) -> &Self::Target {
        &self.tokenizer
    }
}
