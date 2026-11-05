//! Tokenizer wrapper using HuggingFace `tokenizers` crate.

use std::path::Path;

use apxinf_core::{Error, Result};
use minijinja::Environment;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer as HfTokenizer;

/// Chat message for template rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }

    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }
}

/// Configuration loaded from tokenizer_config.json.
#[derive(Debug, Deserialize, Default)]
struct TokenizerConfig {
    chat_template: Option<String>,
    bos_token: Option<SpecialToken>,
    eos_token: Option<SpecialToken>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SpecialToken {
    Text(String),
    Object { content: String },
}

impl SpecialToken {
    fn content(&self) -> &str {
        match self {
            Self::Text(content) | Self::Object { content } => content,
        }
    }
}

/// Wrapper around HuggingFace tokenizer with chat template support.
pub struct Tokenizer {
    inner: HfTokenizer,
    config: TokenizerConfig,
    chat_template: Option<String>,
}

type HfDecodeStream<'a> = tokenizers::DecodeStream<
    'a,
    tokenizers::models::ModelWrapper,
    tokenizers::normalizers::NormalizerWrapper,
    tokenizers::pre_tokenizers::PreTokenizerWrapper,
    tokenizers::processors::PostProcessorWrapper,
    tokenizers::decoders::DecoderWrapper,
>;

/// Incremental text decoding through the tokenizer's existing decoder.
///
/// Special tokens are skipped, as in [`Tokenizer::decode`].
/// Call [`Self::finish`] with the complete output sequence to check and complete the text.
pub struct TextDecodeStream<'a> {
    tokenizer: &'a Tokenizer,
    inner: HfDecodeStream<'a>,
    emitted: String,
}

impl TextDecodeStream<'_> {
    /// Decode one token and return only newly available text.
    ///
    /// `None` means that the decoder has no complete text to emit yet.
    /// Discard the stream after an error.
    pub fn step(&mut self, token: u32) -> Result<Option<String>> {
        let chunk = self
            .inner
            .step(token)
            .map_err(|error| Error::Other(format!("tokenizer stream decode: {error}")))?;
        if let Some(text) = chunk.as_ref() {
            self.emitted.push_str(text);
        }
        Ok(chunk)
    }

    /// Check the emitted prefix and return the remaining text.
    ///
    /// Pass the complete output sequence supplied to [`Self::step`].
    /// The upstream stream has no flush operation. Full decoding preserves its final UTF-8 handling.
    /// A changed prefix returns an error because previously emitted text cannot be withdrawn.
    pub fn finish(self, tokens: &[u32]) -> Result<String> {
        let complete = self.tokenizer.decode(tokens)?;
        complete
            .strip_prefix(&self.emitted)
            .map(str::to_owned)
            .ok_or_else(|| {
                Error::Other("tokenizer stream output differs from complete decode".to_owned())
            })
    }
}

impl Tokenizer {
    /// Load tokenizer from tokenizer.json, optionally loading tokenizer_config.json
    /// and the standard standalone chat_template.jinja from the same directory.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let model_dir = path_ref.parent().unwrap_or_else(|| Path::new("."));
        let config_path = path_ref
            .parent()
            .map(|p| p.join("tokenizer_config.json"))
            .unwrap_or_else(|| path_ref.with_extension("config.json"));
        let tokenizer_json = std::fs::read(path_ref).map_err(|error| {
            Error::Other(format!("tokenizer read {}: {error}", path_ref.display()))
        })?;
        let tokenizer_config_json = if config_path.exists() {
            Some(std::fs::read(&config_path).map_err(|error| {
                Error::Other(format!(
                    "tokenizer config read {}: {error}",
                    config_path.display()
                ))
            })?)
        } else {
            None
        };
        let standalone_path = model_dir.join("chat_template.jinja");
        let needs_standalone = match tokenizer_config_json.as_deref() {
            Some(payload) => serde_json::from_slice::<TokenizerConfig>(payload)
                .map_err(|error| Error::Other(format!("tokenizer config parse: {error}")))?
                .chat_template
                .is_none(),
            None => true,
        };
        let standalone_chat_template = if needs_standalone && standalone_path.exists() {
            Some(std::fs::read(&standalone_path).map_err(|error| {
                Error::Other(format!(
                    "chat template read {}: {error}",
                    standalone_path.display()
                ))
            })?)
        } else {
            None
        };
        Self::from_bytes(
            &tokenizer_json,
            tokenizer_config_json.as_deref(),
            standalone_chat_template.as_deref(),
        )
    }

    /// Load a tokenizer and its optional companion files from already-attested
    /// bytes, without reopening any model path.
    pub fn from_bytes(
        tokenizer_json: &[u8],
        tokenizer_config_json: Option<&[u8]>,
        standalone_chat_template: Option<&[u8]>,
    ) -> Result<Self> {
        let inner = HfTokenizer::from_bytes(tokenizer_json)
            .map_err(|error| Error::Other(format!("tokenizer load: {error}")))?;
        let config = match tokenizer_config_json {
            Some(payload) => serde_json::from_slice(payload)
                .map_err(|error| Error::Other(format!("tokenizer config parse: {error}")))?,
            None => TokenizerConfig::default(),
        };
        let chat_template = load_chat_template_from_bytes(&config, standalone_chat_template)?;
        Ok(Self {
            inner,
            config,
            chat_template,
        })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Other(format!("tokenizer encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, true)
            .map_err(|e| Error::Other(format!("tokenizer decode: {e}")))
    }

    /// Start an incremental decoder that skips special tokens.
    pub fn decode_stream(&self) -> TextDecodeStream<'_> {
        TextDecodeStream {
            tokenizer: self,
            inner: self.inner.decode_stream(true),
            emitted: String::new(),
        }
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get the EOS token ID.
    /// First checks tokenizer_config.json, then falls back to vocabulary search.
    pub fn eos_token_id(&self) -> Option<u32> {
        // Prefer explicit ID from config
        if let Some(id) = self.config.eos_token_id {
            return Some(id);
        }

        // Try to find by token name from config
        if let Some(token) = &self.config.eos_token {
            if let Some(&id) = self.inner.get_vocab(true).get(token.content()) {
                return Some(id);
            }
        }

        // Fallback: search vocabulary for common EOS tokens
        self.inner
            .get_vocab(true)
            .iter()
            .find(|(token, _)| {
                *token == "</s>" || *token == "<|eot_id|>" || *token == "<|end_of_text|>"
            })
            .map(|(_, &id)| id)
    }

    /// Get the BOS (beginning-of-sequence) token ID.
    /// First checks tokenizer_config.json, then falls back to vocabulary search.
    pub fn bos_token_id(&self) -> Option<u32> {
        // Prefer explicit ID from config
        if let Some(id) = self.config.bos_token_id {
            return Some(id);
        }

        // Try to find by token name from config
        if let Some(token) = &self.config.bos_token {
            if let Some(&id) = self.inner.get_vocab(true).get(token.content()) {
                return Some(id);
            }
        }

        // Fallback: search vocabulary for common BOS tokens
        self.inner
            .get_vocab(true)
            .iter()
            .find(|(token, _)| *token == "<s>" || *token == "<|begin_of_text|>")
            .map(|(_, &id)| id)
    }

    /// Check if a chat template is available.
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    /// Apply chat template to messages, returning formatted prompt string.
    ///
    /// Requires either tokenizer_config.json with a `chat_template` field or
    /// the standard sibling chat_template.jinja file.
    /// Uses minijinja to render the Jinja2 template.
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        let template_str = self.chat_template.as_ref().ok_or_else(|| {
            Error::Other(
                "no chat template available (missing tokenizer_config.json chat_template and chat_template.jinja)"
                    .to_string(),
            )
        })?;

        // Create environment and template on demand
        let mut env = Environment::new();
        // Hugging Face chat templates are Python-flavoured Jinja and commonly
        // call methods such as `startswith`, `endswith`, `split`, and `rstrip`.
        // MiniJinja deliberately does not expose those methods by default.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.set_keep_trailing_newline(true);
        env.add_template("chat", template_str)
            .map_err(|e| Error::Other(format!("template error: {e}")))?;

        let tmpl = env
            .get_template("chat")
            .map_err(|e| Error::Other(format!("template error: {e}")))?;

        // Build template context
        let bos = self
            .config
            .bos_token
            .as_ref()
            .map(|token| token.content().to_owned())
            .or_else(|| {
                self.inner
                    .get_vocab(true)
                    .iter()
                    .find(|(t, _)| *t == "<s>" || *t == "<|begin_of_text|>")
                    .map(|(t, _)| t.clone())
            })
            .unwrap_or_default();

        let eos = self
            .config
            .eos_token
            .as_ref()
            .map(|token| token.content().to_owned())
            .or_else(|| {
                self.inner
                    .get_vocab(true)
                    .iter()
                    .find(|(t, _)| *t == "</s>" || *t == "<|eot_id|>" || *t == "<|end_of_text|>")
                    .map(|(t, _)| t.clone())
            })
            .unwrap_or_default();

        // Create context as serde Value (map)
        let context = serde_json::json!({
            "messages": messages,
            "bos_token": bos,
            "eos_token": eos,
            "add_generation_prompt": true,
            // Qwen3.5-0.8B is a non-thinking model. Keeping this explicit also
            // makes the rendered suffix deterministic across Jinja engines.
            "enable_thinking": false,
        });

        // Preserve the template's output byte-for-byte. Consecutive newlines
        // are semantically significant in Qwen3.5's non-thinking generation
        // suffix and therefore must not be collapsed after rendering.
        tmpl.render(context)
            .map_err(|e| Error::Other(format!("template render error: {e}")))
    }

    /// Encode messages using chat template.
    /// Convenience method that applies template and encodes the result.
    pub fn encode_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let prompt = self.apply_chat_template(messages)?;
        println!("Formatted prompt:\n{}", prompt);
        self.encode(&prompt)
    }
}

#[cfg(test)]
fn load_chat_template(model_dir: &Path, config: &TokenizerConfig) -> Result<Option<String>> {
    if let Some(template) = config.chat_template.as_ref() {
        if template.is_empty() {
            return Err(Error::Other(
                "tokenizer_config.json chat_template must not be empty".to_string(),
            ));
        }
        return Ok(Some(template.clone()));
    }

    let standalone = model_dir.join("chat_template.jinja");
    if !standalone.exists() {
        return Ok(None);
    }
    let template = std::fs::read_to_string(&standalone).map_err(|error| {
        Error::Other(format!(
            "chat template read {}: {error}",
            standalone.display()
        ))
    })?;
    if template.is_empty() {
        return Err(Error::Other(format!(
            "chat template {} must not be empty",
            standalone.display()
        )));
    }
    Ok(Some(template))
}

fn load_chat_template_from_bytes(
    config: &TokenizerConfig,
    standalone: Option<&[u8]>,
) -> Result<Option<String>> {
    if let Some(template) = config.chat_template.as_ref() {
        if template.is_empty() {
            return Err(Error::Other(
                "tokenizer_config.json chat_template must not be empty".to_string(),
            ));
        }
        return Ok(Some(template.clone()));
    }
    let Some(bytes) = standalone else {
        return Ok(None);
    };
    let template = std::str::from_utf8(bytes)
        .map_err(|error| Error::Other(format!("chat template is not UTF-8: {error}")))?;
    if template.is_empty() {
        return Err(Error::Other(
            "chat_template.jinja must not be empty".to_string(),
        ));
    }
    Ok(Some(template.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn tokenizer_with_template(template: &str) -> Tokenizer {
        Tokenizer {
            inner: HfTokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            config: TokenizerConfig::default(),
            chat_template: Some(template.to_owned()),
        }
    }

    fn tokenizer_with_byte_tokens() -> Tokenizer {
        let vocabulary = [
            ("<unk>", 0),
            ("<0x41>", 1),
            ("<0x42>", 2),
            ("<0x20>", 3),
            ("<0xE4>", 4),
            ("<0xBD>", 5),
            ("<0xA0>", 6),
            ("<0xE5>", 7),
            ("<0xA5>", 8),
            ("<eos>", 9),
            ("A", 10),
        ]
        .into_iter()
        .map(|(token, id)| (token.to_owned(), id))
        .collect();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocabulary)
            .unk_token("<unk>".to_owned())
            .build()
            .unwrap();
        let mut inner = HfTokenizer::new(model);
        inner.with_decoder(Some(
            tokenizers::decoders::byte_fallback::ByteFallback::new(),
        ));
        inner.add_special_tokens(&[tokenizers::AddedToken::from("<eos>", true)]);
        Tokenizer {
            inner,
            config: TokenizerConfig::default(),
            chat_template: None,
        }
    }

    #[test]
    fn streamed_ascii_matches_complete_decode_without_repeated_prefixes() {
        let tokenizer = tokenizer_with_byte_tokens();
        let tokens = [1, 3, 2];
        let mut stream = tokenizer.decode_stream();
        let mut text = String::new();
        for (token, expected) in tokens.into_iter().zip(["A", " ", "B"]) {
            let chunk = stream.step(token).unwrap().unwrap();
            assert_eq!(chunk, expected);
            text.push_str(&chunk);
        }
        let tail = stream.finish(&tokens).unwrap();
        assert!(tail.is_empty());
        text.push_str(&tail);
        assert_eq!(text, "A B");
        assert_eq!(text, tokenizer.decode(&tokens).unwrap());
    }

    #[test]
    fn streamed_chinese_waits_for_complete_utf8_and_skips_special_tokens() {
        let tokenizer = tokenizer_with_byte_tokens();
        let tokens = [4, 5, 6, 7, 8, 5, 9];
        let mut stream = tokenizer.decode_stream();
        let mut text = String::new();
        for (token, expected) in
            tokens
                .into_iter()
                .zip([None, None, Some("你"), None, None, Some("好"), None])
        {
            let chunk = stream.step(token).unwrap();
            assert_eq!(chunk.as_deref(), expected);
            if let Some(chunk) = chunk {
                text.push_str(&chunk);
            }
        }
        text.push_str(&stream.finish(&tokens).unwrap());
        assert_eq!(text, "你好");
        assert_eq!(text, tokenizer.decode(&tokens).unwrap());
    }

    #[test]
    fn stream_finish_preserves_incomplete_byte_suffix_like_complete_decode() {
        let tokenizer = tokenizer_with_byte_tokens();
        let tokens = [10, 4];
        let mut stream = tokenizer.decode_stream();
        let first = stream.step(tokens[0]).unwrap().unwrap();
        assert!(stream.step(tokens[1]).unwrap().is_none());
        let tail = stream.finish(&tokens).unwrap();
        assert!(!tail.is_empty());
        assert_eq!(first + &tail, tokenizer.decode(&tokens).unwrap());
    }

    #[test]
    fn stream_finish_rejects_a_different_complete_prefix() {
        let tokenizer = tokenizer_with_byte_tokens();
        let mut stream = tokenizer.decode_stream();
        assert_eq!(stream.step(1).unwrap().as_deref(), Some("A"));
        assert!(stream.finish(&[2]).is_err());
    }

    #[test]
    fn stream_finish_rejects_decoder_changes_to_emitted_text() {
        let tokenizer = tokenizer_with_byte_tokens();
        let tokens = [1, 4];
        let mut stream = tokenizer.decode_stream();
        assert_eq!(stream.step(tokens[0]).unwrap().as_deref(), Some("A"));
        assert!(stream.step(tokens[1]).unwrap().is_none());
        assert!(!tokenizer.decode(&tokens).unwrap().starts_with('A'));
        assert!(stream.finish(&tokens).is_err());
    }

    #[test]
    fn empty_stream_and_special_only_stream_finish_without_text() {
        let tokenizer = tokenizer_with_byte_tokens();
        assert_eq!(tokenizer.decode_stream().finish(&[]).unwrap(), "");
        let mut stream = tokenizer.decode_stream();
        assert!(stream.step(9).unwrap().is_none());
        assert_eq!(stream.finish(&[9]).unwrap(), "");
    }

    #[test]
    fn test_chat_message_constructors() {
        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, "user");
        assert_eq!(user.content, "Hello");

        let assistant = ChatMessage::assistant("Hi there");
        assert_eq!(assistant.role, "assistant");

        let system = ChatMessage::system("You are helpful");
        assert_eq!(system.role, "system");
    }

    #[test]
    fn python_string_methods_and_newlines_match_hf_templates() {
        let tokenizer = tokenizer_with_template(
            "{%- set content = messages[0].content -%}\
             {%- if content.startswith('tool:') and content.endswith(':end') -%}\
             {{- '<think>\\n\\n</think>\\n\\n' -}}\
             {%- endif -%}",
        );

        let rendered = tokenizer
            .apply_chat_template(&[ChatMessage::user("tool:value:end")])
            .unwrap();
        assert_eq!(rendered, "<think>\n\n</think>\n\n");
    }

    #[test]
    fn standalone_chat_template_is_used_when_config_has_no_embedded_template() {
        let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "apxinf-tokenizer-chat-template-{}-{serial}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        std::fs::write(
            root.join("chat_template.jinja"),
            "{{ messages[0].content }}\n",
        )
        .unwrap();

        let loaded = load_chat_template(&root, &TokenizerConfig::default()).unwrap();
        assert_eq!(loaded.as_deref(), Some("{{ messages[0].content }}\n"));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn embedded_chat_template_takes_precedence_over_standalone_file() {
        let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "apxinf-tokenizer-embedded-template-{}-{serial}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("chat_template.jinja"), "standalone").unwrap();
        let config = TokenizerConfig {
            chat_template: Some("embedded".to_string()),
            ..TokenizerConfig::default()
        };

        let loaded = load_chat_template(&root, &config).unwrap();
        assert_eq!(loaded.as_deref(), Some("embedded"));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn from_bytes_is_independent_of_later_path_rebinding() {
        let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "apxinf-tokenizer-owned-bytes-{}-{serial}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        let config_path = root.join("tokenizer_config.json");
        std::fs::write(
            &config_path,
            br#"{"chat_template":"original:{{ messages[0].content }}"}"#,
        )
        .unwrap();
        let captured_config = std::fs::read(&config_path).unwrap();
        std::fs::rename(&config_path, root.join("original.json")).unwrap();
        std::fs::write(
            &config_path,
            br#"{"chat_template":"replacement:{{ messages[0].content }}"}"#,
        )
        .unwrap();
        let tokenizer_json = HfTokenizer::new(tokenizers::models::wordlevel::WordLevel::default())
            .to_string(false)
            .unwrap();

        let tokenizer =
            Tokenizer::from_bytes(tokenizer_json.as_bytes(), Some(&captured_config), None).unwrap();
        assert_eq!(
            tokenizer
                .apply_chat_template(&[ChatMessage::user("hello")])
                .unwrap(),
            "original:hello"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn special_tokens_accept_hugging_face_string_and_object_forms() {
        let config: TokenizerConfig = serde_json::from_str(
            r#"{
                "bos_token": "<s>",
                "eos_token": {"content": "</s>", "lstrip": false}
            }"#,
        )
        .unwrap();
        assert_eq!(config.bos_token.unwrap().content(), "<s>");
        assert_eq!(config.eos_token.unwrap().content(), "</s>");
    }
}
