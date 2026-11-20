//! Speech recognition core: Qwen3-ASR running on llama.cpp's multimodal (mtmd) audio path.
//!
//! A model is a pair of GGUF files: the text decoder (`Qwen3-ASR-*.gguf`) and the audio
//! encoder/projector (`mmproj-Qwen3-ASR-*.gguf`).

use std::ffi::CStr;
use std::num::NonZeroU32;
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText, mtmd_default_marker,
};
use llama_cpp_2::sampling::LlamaSampler;
use thiserror::Error;

pub const SAMPLE_RATE: usize = 16_000;

/// Qwen3-ASR's audio encoder needs a little audio to produce any frames; shorter clips are
/// left-padded with silence.
const MIN_AUDIO_SAMPLES: usize = SAMPLE_RATE / 2;
/// Context window for one utterance: the prompt, the audio embeddings and the transcript.
/// The app sends at most ~30 s per request; Qwen3-ASR uses ~13 audio tokens per second.
const N_CTX: u32 = 4096;
const N_BATCH: u32 = 2048;
/// Separates the language tag from the transcript in Qwen3-ASR's output
/// (`language Japanese<asr_text>...`).
const ASR_TEXT_TAG: &str = "<asr_text>";

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static LAST_LOG_EMITTED: AtomicBool = AtomicBool::new(false);

/// Routes llama.cpp / ggml / mtmd logs to the `log` facade: warnings and errors at their own
/// level, everything else at `trace` (llama.cpp is very chatty at INFO).
unsafe extern "C" fn llama_log_callback(level: u32, text: *const c_char, _user_data: *mut c_void) {
    if text.is_null() {
        return;
    }
    let msg = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    // ggml_log_level: NONE=0, DEBUG=1, INFO=2, WARN=3, ERROR=4, CONT=5. CONT continues the
    // previous message and inherits its level.
    let level = match level {
        3 => log::Level::Warn,
        4 => log::Level::Error,
        5 => {
            if LAST_LOG_EMITTED.load(Ordering::Relaxed) {
                log::Level::Warn
            } else {
                log::Level::Trace
            }
        }
        _ => log::Level::Trace,
    };
    LAST_LOG_EMITTED.store(level <= log::Level::Warn, Ordering::Relaxed);
    log::log!(target: "llama_cpp", level, "{}", msg.trim_end());
}

fn backend() -> Result<&'static LlamaBackend, AsrError> {
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    // SAFETY: the callback is a plain function that only reads the C string it is given.
    unsafe {
        llama_cpp_sys_2::llama_log_set(Some(llama_log_callback), std::ptr::null_mut());
        llama_cpp_sys_2::mtmd_helper_log_set(Some(llama_log_callback), std::ptr::null_mut());
        llama_cpp_sys_2::ggml_log_set(Some(llama_log_callback), std::ptr::null_mut());
    }
    let backend =
        LlamaBackend::init().map_err(|e| AsrError::InitializationFailed(e.to_string()))?;
    Ok(BACKEND.get_or_init(|| backend))
}

#[derive(Error, Debug)]
pub enum AsrError {
    #[error("Failed to initialize model: {0}")]
    InitializationFailed(String),
    #[error("Failed to process audio: {0}")]
    ProcessingFailed(String),
}

#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// ISO 639-1 code (e.g. "ja"). `None`, "auto" or an unsupported code lets the model detect
    /// the language.
    pub language: Option<String>,
    /// Optional context text (domain terms, names) placed in the system turn to bias
    /// recognition.
    pub context: Option<String>,
}

pub struct AsrModel {
    model: LlamaModel,
    /// Encoding is not thread-safe, so the multimodal context is used under a lock.
    mtmd: Mutex<MtmdContext>,
}

impl AsrModel {
    /// Loads the model with Metal acceleration.
    pub fn load<P: AsRef<Path>, Q: AsRef<Path>>(
        model_path: P,
        mmproj_path: Q,
    ) -> Result<Self, AsrError> {
        Self::load_with_device(model_path, mmproj_path, true)
    }

    /// Loads the model, on the GPU (Metal) or CPU only.
    pub fn load_with_device<P: AsRef<Path>, Q: AsRef<Path>>(
        model_path: P,
        mmproj_path: Q,
        use_gpu: bool,
    ) -> Result<Self, AsrError> {
        let backend = backend()?;
        let model_params = if use_gpu {
            LlamaModelParams::default().with_n_gpu_layers(u32::MAX)
        } else {
            // An empty device list keeps llama.cpp from initializing Metal at all.
            LlamaModelParams::default()
                .with_n_gpu_layers(0)
                .with_devices(&[])
                .map_err(|e| AsrError::InitializationFailed(e.to_string()))?
        };
        let model = LlamaModel::load_from_file(backend, model_path.as_ref(), &model_params)
            .map_err(|e| AsrError::InitializationFailed(format!("text model: {e}")))?;

        let mmproj_path = mmproj_path
            .as_ref()
            .to_str()
            .ok_or_else(|| AsrError::InitializationFailed("invalid mmproj path".into()))?;
        let mtmd_params = MtmdContextParams {
            use_gpu,
            print_timings: false,
            n_threads: num_threads(),
            ..MtmdContextParams::default()
        };
        let mtmd = MtmdContext::init_from_file(mmproj_path, &model, &mtmd_params)
            .map_err(|e| AsrError::InitializationFailed(format!("mmproj: {e}")))?;
        if !mtmd.support_audio() {
            return Err(AsrError::InitializationFailed(
                "mmproj does not support audio input".into(),
            ));
        }

        Ok(Self {
            model,
            mtmd: Mutex::new(mtmd),
        })
    }

    /// Transcribes 16 kHz mono f32 PCM and returns the recognized text.
    pub fn transcribe(
        &self,
        audio: &[f32],
        options: &TranscribeOptions,
    ) -> Result<String, AsrError> {
        let backend = backend()?;
        let failed = |stage: &str, e: &dyn std::fmt::Display| {
            AsrError::ProcessingFailed(format!("{stage}: {e}"))
        };

        let audio = pad_to_min_length(audio);
        let language = options.language.as_deref().and_then(qwen3_language_name);
        let prompt = build_prompt(language, options.context.as_deref());

        let mut mtmd = self
            .mtmd
            .lock()
            .map_err(|_| AsrError::ProcessingFailed("mtmd lock poisoned".into()))?;

        let bitmap = MtmdBitmap::from_audio_data(&audio).map_err(|e| failed("audio", &e))?;
        let chunks = mtmd
            .tokenize(
                MtmdInputText {
                    text: prompt,
                    add_special: true,
                    parse_special: true,
                },
                &[&bitmap],
            )
            .map_err(|e| failed("tokenize", &e))?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(N_CTX))
            .with_n_batch(N_BATCH)
            .with_n_ubatch(N_BATCH)
            .with_n_threads(num_threads())
            .with_n_threads_batch(num_threads());
        let mut ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| failed("context", &e))?;

        let prompt_end = chunks
            .eval_chunks(&mut mtmd, &mut ctx, 0, 0, N_BATCH as i32, true)
            .map_err(|e| failed("eval", &e))?;
        drop(mtmd);

        let generation_end =
            prompt_end + max_new_tokens(audio.len()).min(N_CTX as i32 - prompt_end);
        let vocab = self.model.vocab();
        let mut sampler = LlamaSampler::greedy();
        let mut batch = LlamaBatch::new(1, 1);
        let mut output = Vec::new();
        for pos in prompt_end..generation_end {
            let token = sampler.sample(&ctx, -1);
            if vocab.is_eog(token) {
                break;
            }
            output.extend_from_slice(&vocab.token_to_piece(token, false, None));

            batch.clear();
            batch
                .add(token, pos, &[0], true)
                .map_err(|e| failed("batch", &e))?;
            ctx.decode(&mut batch).map_err(|e| failed("decode", &e))?;
        }

        Ok(extract_transcript(&String::from_utf8_lossy(&output)))
    }
}

/// Qwen3-ASR chat prompt. When the language is known it is forced by pre-filling the
/// assistant turn with the language tag, so the model emits only the transcript.
fn build_prompt(language: Option<&str>, context: Option<&str>) -> String {
    let marker = mtmd_default_marker();
    let context = context.unwrap_or("");
    let mut prompt = format!(
        "<|im_start|>system\n{context}<|im_end|>\n\
         <|im_start|>user\n{marker}<|im_end|>\n\
         <|im_start|>assistant\n"
    );
    if let Some(language) = language {
        prompt.push_str("language ");
        prompt.push_str(language);
        prompt.push_str(ASR_TEXT_TAG);
    }
    prompt
}

/// Drops the `language X<asr_text>` header the model emits when the language was not forced.
fn extract_transcript(raw: &str) -> String {
    let text = match raw.find(ASR_TEXT_TAG) {
        Some(pos) => &raw[pos + ASR_TEXT_TAG.len()..],
        None => raw,
    };
    text.trim().to_string()
}

/// Upper bound on generated tokens, proportional to the audio length. Guards against
/// repetition loops running to the end of the context.
fn max_new_tokens(samples: usize) -> i32 {
    let seconds = samples as f32 / SAMPLE_RATE as f32;
    (seconds * 20.0) as i32 + 32
}

fn pad_to_min_length(audio: &[f32]) -> Vec<f32> {
    if audio.len() >= MIN_AUDIO_SAMPLES {
        return audio.to_vec();
    }
    let mut padded = vec![0.0; MIN_AUDIO_SAMPLES - audio.len()];
    padded.extend_from_slice(audio);
    padded
}

/// Maps an ISO 639-1 code to the language name Qwen3-ASR expects in its prompt.
pub fn qwen3_language_name(code: &str) -> Option<&'static str> {
    let name = match code.trim().to_ascii_lowercase().as_str() {
        "ja" => "Japanese",
        "en" => "English",
        "zh" => "Chinese",
        "yue" => "Cantonese",
        "ko" => "Korean",
        "ar" => "Arabic",
        "de" => "German",
        "fr" => "French",
        "es" => "Spanish",
        "pt" => "Portuguese",
        "id" => "Indonesian",
        "it" => "Italian",
        "ru" => "Russian",
        "th" => "Thai",
        "vi" => "Vietnamese",
        "tr" => "Turkish",
        "hi" => "Hindi",
        "ms" => "Malay",
        "nl" => "Dutch",
        "sv" => "Swedish",
        "da" => "Danish",
        "fi" => "Finnish",
        "pl" => "Polish",
        "cs" => "Czech",
        "fil" | "tl" => "Filipino",
        "fa" => "Persian",
        "el" => "Greek",
        "ro" => "Romanian",
        "hu" => "Hungarian",
        "mk" => "Macedonian",
        _ => return None,
    };
    Some(name)
}

fn num_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
}

pub fn convert_pcm_to_f32(pcm_data: &[i16]) -> Vec<f32> {
    pcm_data
        .iter()
        .map(|&sample| sample as f32 / 32768.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_language_prefills_assistant_turn() {
        let prompt = build_prompt(Some("Japanese"), None);
        assert!(prompt.ends_with("<|im_start|>assistant\nlanguage Japanese<asr_text>"));
        assert!(prompt.contains(mtmd_default_marker()));
    }

    #[test]
    fn auto_language_leaves_assistant_turn_open() {
        let prompt = build_prompt(None, Some("用語: Tauri"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
        assert!(prompt.starts_with("<|im_start|>system\n用語: Tauri<|im_end|>"));
    }

    #[test]
    fn transcript_header_is_stripped() {
        assert_eq!(
            extract_transcript("language Japanese<asr_text>こんにちは。"),
            "こんにちは。"
        );
        assert_eq!(extract_transcript(" こんにちは。 "), "こんにちは。");
    }

    #[test]
    fn language_codes_map_to_names() {
        assert_eq!(qwen3_language_name("ja"), Some("Japanese"));
        assert_eq!(qwen3_language_name("JA"), Some("Japanese"));
        assert_eq!(qwen3_language_name("auto"), None);
    }

    #[test]
    fn short_audio_is_left_padded() {
        let padded = pad_to_min_length(&[1.0; 10]);
        assert_eq!(padded.len(), MIN_AUDIO_SAMPLES);
        assert_eq!(padded[MIN_AUDIO_SAMPLES - 1], 1.0);
        assert_eq!(padded[0], 0.0);
    }
}
