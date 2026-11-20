pub const VAD_SAMPLE_RATE: u32 = 16_000;
pub const VAD_CHUNK_SIZE: usize = 500;
pub const DEFAULT_VAD_THRESHOLD: f32 = 0.1;
pub const DEFAULT_PARTIAL_TRANSCRIPT_INTERVAL_SAMPLES: usize = 2 * VAD_SAMPLE_RATE as usize;
pub const SILENCE_TIMEOUT_SAMPLES: usize = 2 * VAD_SAMPLE_RATE as usize;
pub const VAD_PRE_BUFFER_MS: usize = 200;
pub const VAD_POST_BUFFER_MS: usize = 200;
pub const VAD_PRE_BUFFER_SAMPLES: usize = (VAD_SAMPLE_RATE as usize * VAD_PRE_BUFFER_MS) / 1000;
pub const VAD_POST_BUFFER_SAMPLES: usize = (VAD_SAMPLE_RATE as usize * VAD_POST_BUFFER_MS) / 1000;
/// Longest utterance sent to the model in one request. A session is finalized when it reaches
/// this length, which also bounds the cost of re-transcribing it for partial results.
pub const SESSION_MAX_SECONDS: usize = 30;
pub const SESSION_MAX_SAMPLES: usize = SESSION_MAX_SECONDS * VAD_SAMPLE_RATE as usize;
