use std::borrow::Cow;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use thiserror::Error;
use whisper_rs::{
    FullParams, GGMLLogLevel, SamplingStrategy, WhisperContext as WhisperRsContext,
    WhisperContextParameters,
};

const WHISPER_SAMPLE_RATE: usize = 16_000;
const MIN_AUDIO_DURATION_S: usize = 2;
const MIN_AUDIO_SAMPLES: usize = WHISPER_SAMPLE_RATE * MIN_AUDIO_DURATION_S;
static WHISPER_LOG_INIT: Once = Once::new();
static LAST_LOG_EMITTED: AtomicBool = AtomicBool::new(false);

unsafe extern "C" fn whisper_log_filter_callback(
    level: u32,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    if text.is_null() {
        return;
    }

    let msg = unsafe { CStr::from_ptr(text) }.to_string_lossy();

    // Suppress verbose decoder traces such as:
    // "whisper_full_with_state: id = ..."
    let suppressed = msg.contains("whisper_full_with_state");

    // Keep only warning/error messages from whisper.cpp.
    // ggml_log_level (whisper-rs-sys 0.15): NONE=0, DEBUG=1, INFO=2, WARN=3, ERROR=4, CONT=5.
    // CONT continues the previous message, so it inherits that message's verdict
    // (including suppression of trace lines).
    let emit = match GGMLLogLevel::from(level) {
        GGMLLogLevel::Cont => !suppressed && LAST_LOG_EMITTED.load(Ordering::Relaxed),
        other => {
            let keep = !suppressed && matches!(other, GGMLLogLevel::Warn | GGMLLogLevel::Error);
            LAST_LOG_EMITTED.store(keep, Ordering::Relaxed);
            keep
        }
    };
    if emit {
        eprint!("{msg}");
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WhisperParams {
    pub audio_ctx: i32,
    pub temperature: f32,
}

impl Default for WhisperParams {
    fn default() -> Self {
        Self {
            audio_ctx: 1500,
            temperature: 0.0,
        }
    }
}

impl WhisperParams {
    pub fn clamped(self) -> Self {
        Self {
            audio_ctx: self.audio_ctx.clamp(64, 1500),
            temperature: self.temperature.clamp(0.0, 1.0),
        }
    }
}

#[derive(Error, Debug)]
pub enum WhisperError {
    #[error("Failed to initialize context: {0}")]
    InitializationFailed(String),
    #[error("Failed to process audio: {0}")]
    ProcessingFailed(String),
    #[error("Invalid model path")]
    InvalidModelPath,
}

pub struct WhisperContext {
    ctx: WhisperRsContext,
    params: WhisperParams,
}

#[derive(Debug, Clone)]
pub struct TranscribedSegment {
    pub text: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub words: Vec<TranscribedWord>,
}

#[derive(Debug, Clone)]
pub struct TranscribedWord {
    pub text: String,
    pub start_ms: i64,
    pub end_ms: i64,
}

impl WhisperContext {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, WhisperError> {
        WHISPER_LOG_INIT.call_once(|| unsafe {
            whisper_rs::set_log_callback(Some(whisper_log_filter_callback), std::ptr::null_mut());
        });

        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(true).flash_attn(true);

        let ctx = WhisperRsContext::new_with_params(
            model_path
                .as_ref()
                .to_str()
                .ok_or(WhisperError::InvalidModelPath)?,
            ctx_params,
        )
        .map_err(|e| WhisperError::InitializationFailed(e.to_string()))?;

        Ok(Self {
            ctx,
            params: WhisperParams::default(),
        })
    }

    pub fn transcribe(&self, audio_data: &[f32]) -> Result<String, WhisperError> {
        self.transcribe_with_language(audio_data, "ja")
    }

    pub fn set_params(&mut self, params: WhisperParams) {
        self.params = params.clamped();
    }

    pub fn params(&self) -> WhisperParams {
        self.params
    }

    pub fn transcribe_segments_with_language(
        &self,
        audio_data: &[f32],
        language: &str,
    ) -> Result<Vec<TranscribedSegment>, WhisperError> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| WhisperError::ProcessingFailed(e.to_string()))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_translate(false);
        params.set_language(Some(language));
        params.set_n_threads(num_cpus());
        params.set_single_segment(false);
        params.set_audio_ctx(self.params.audio_ctx);
        params.set_temperature(self.params.temperature);
        params.set_token_timestamps(true);
        params.set_thold_pt(0.01);
        params.set_thold_ptsum(0.01);
        params.set_max_len(1);
        params.set_split_on_word(true);

        let audio_for_inference = ensure_min_audio_duration(audio_data);
        let inference_start = Instant::now();
        state
            .full(params, audio_for_inference.as_ref())
            .map_err(|e| WhisperError::ProcessingFailed(e.to_string()))?;
        let _inference_elapsed = inference_start.elapsed();

        let num_segments = state.full_n_segments();

        let mut segments = Vec::with_capacity(num_segments.max(0) as usize);
        for i in 0..num_segments {
            let segment = state.get_segment(i).ok_or_else(|| {
                WhisperError::ProcessingFailed(format!("segment {i} out of bounds"))
            })?;
            let text = segment
                .to_str_lossy()
                .map_err(|e| WhisperError::ProcessingFailed(e.to_string()))?
                .into_owned();
            let words = collect_words(&segment)?;
            segments.push(TranscribedSegment {
                text,
                start_ms: segment.start_timestamp() * 10,
                end_ms: segment.end_timestamp() * 10,
                words,
            });
        }

        Ok(segments)
    }

    pub fn transcribe_with_language(
        &self,
        audio_data: &[f32],
        language: &str,
    ) -> Result<String, WhisperError> {
        let segments = self.transcribe_segments_with_language(audio_data, language)?;
        let mut full_text = String::new();
        for segment in segments {
            full_text.push_str(&segment.text);
        }
        Ok(full_text)
    }
}

fn collect_words(
    segment: &whisper_rs::WhisperSegment<'_>,
) -> Result<Vec<TranscribedWord>, WhisperError> {
    let token_count = segment.n_tokens();

    let mut words = Vec::with_capacity(token_count.max(0) as usize);
    for token_idx in 0..token_count {
        let Some(token) = segment.get_token(token_idx) else {
            continue;
        };
        let token_text = token
            .to_str_lossy()
            .map_err(|e| WhisperError::ProcessingFailed(e.to_string()))?;
        let token_data = token.token_data();

        if token_data.t0 < 0 || token_data.t1 <= token_data.t0 {
            continue;
        }

        let trimmed = token_text.trim();
        if trimmed.is_empty() {
            continue;
        }

        words.push(TranscribedWord {
            text: trimmed.to_string(),
            start_ms: token_data.t0 * 10,
            end_ms: token_data.t1 * 10,
        });
    }

    Ok(words)
}

fn ensure_min_audio_duration(audio_data: &[f32]) -> Cow<'_, [f32]> {
    if audio_data.len() >= MIN_AUDIO_SAMPLES {
        Cow::Borrowed(audio_data)
    } else {
        let pad_len = MIN_AUDIO_SAMPLES - audio_data.len();
        let mut padded = Vec::with_capacity(MIN_AUDIO_SAMPLES);
        padded.resize(pad_len, 0.0);
        padded.extend_from_slice(audio_data);
        Cow::Owned(padded)
    }
}

fn num_cpus() -> i32 {
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
