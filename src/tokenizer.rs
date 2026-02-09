use std::path::Path;

/// Wrapper around HuggingFace tokenizers for Qwen3.
pub struct Qwen3Tokenizer {
    tokenizer: tokenizers::Tokenizer,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl Qwen3Tokenizer {
    /// Load the tokenizer from a `tokenizer.json` file.
    pub fn new(tokenizer_path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            tokenizer,
            bos_token_id: 151643,
            eos_token_id: 151645,
        })
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

    /// Decode a single token ID to text.
    pub fn decode_token(&self, token: u32) -> String {
        self.tokenizer
            .decode(&[token], false)
            .expect("Failed to decode token")
    }

    /// Get the beginning-of-sequence token ID.
    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    /// Get the end-of-sequence token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Format a user message using Qwen3's chat template.
    pub fn apply_chat_template(&self, system_prompt: &str, user_message: &str) -> String {
        format!(
            "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        )
    }
}
