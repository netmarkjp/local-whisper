use asr_core::{WhisperContext, WhisperParams};
use log::info;
use once_cell::sync::OnceCell;
use parking_lot::Mutex as ParkingMutex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tokio::fs;
use tokio::io::AsyncWriteExt;

use crate::audio::constants::{self, VAD_SAMPLE_RATE};
use crate::audio::state::try_recording_state;
use crate::APP_HANDLE;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
    pub size: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RemoteModelStatus {
    pub id: String,
    pub name: String,
    pub filename: String,
    pub size: u64,
    pub description: String,
    pub installed: bool,
    pub path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct WhisperParamsConfig {
    #[serde(rename = "audioCtx")]
    pub audio_ctx: i32,
    pub temperature: f32,
}

impl From<WhisperParams> for WhisperParamsConfig {
    fn from(params: WhisperParams) -> Self {
        Self {
            audio_ctx: params.audio_ctx,
            temperature: params.temperature,
        }
    }
}

impl From<WhisperParamsConfig> for WhisperParams {
    fn from(config: WhisperParamsConfig) -> Self {
        WhisperParams {
            audio_ctx: config.audio_ctx,
            temperature: config.temperature,
        }
        .clamped()
    }
}

#[derive(Debug)]
pub struct RemoteModel {
    pub id: &'static str,
    pub name: &'static str,
    pub filename: &'static str,
    pub size: u64,
    pub description: &'static str,
    /// Expected SHA-256 (lowercase hex) of the file at `HF_REVISION`.
    pub sha256: &'static str,
}

pub static WHISPER_CTX: OnceCell<Arc<Mutex<Option<WhisperContext>>>> = OnceCell::new();
pub static WHISPER_PARAMS: OnceCell<Arc<ParkingMutex<WhisperParams>>> = OnceCell::new();

/// Model ids whose installation is currently running. Concurrent installs of the same model
/// would share the temp file, so a second request is rejected while the first is in flight.
static MODEL_INSTALLS_IN_FLIGHT: ParkingMutex<BTreeSet<&'static str>> =
    ParkingMutex::new(BTreeSet::new());

/// Removes the model id from `MODEL_INSTALLS_IN_FLIGHT` when dropped (on every exit path).
struct InstallGuard(&'static str);

impl InstallGuard {
    fn acquire(model_id: &'static str) -> Option<Self> {
        MODEL_INSTALLS_IN_FLIGHT
            .lock()
            .insert(model_id)
            .then_some(Self(model_id))
    }
}

impl Drop for InstallGuard {
    fn drop(&mut self) {
        MODEL_INSTALLS_IN_FLIGHT.lock().remove(self.0);
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelInstallProgressEvent {
    #[serde(rename = "modelId")]
    model_id: String,
    filename: String,
    #[serde(rename = "downloadedBytes")]
    downloaded_bytes: u64,
    #[serde(rename = "totalBytes")]
    total_bytes: u64,
    percent: f32,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

/// Pinned commit of https://huggingface.co/ggerganov/whisper.cpp (committed 2024-10-29).
///
/// To update: fetch the new commit SHA from
/// `https://huggingface.co/api/models/ggerganov/whisper.cpp` (`sha`), then for each model
/// run `curl -s -X POST -d 'paths=<file>&expand=true'
/// https://huggingface.co/api/models/ggerganov/whisper.cpp/paths-info/<sha>` and copy
/// `lfs.oid` (SHA-256) and `size` into `REMOTE_MODELS`. Update this constant and all
/// entries together; never download the files just to compute hashes from untrusted copies.
const HF_REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";

pub const REMOTE_MODELS: &[RemoteModel] = &[
    RemoteModel {
        id: "base",
        name: "Base",
        filename: "ggml-base.bin",
        size: 147_951_465,
        description: "英語・多言語兼用 / 約 148 MB",
        sha256: "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
    },
    RemoteModel {
        id: "small",
        name: "Small",
        filename: "ggml-small.bin",
        size: 487_601_967,
        description: "中規模モデル / 約 488 MB",
        sha256: "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
    },
    RemoteModel {
        id: "medium",
        name: "Medium",
        filename: "ggml-medium.bin",
        size: 1_533_763_059,
        description: "高精度モデル / 約 1.5 GB",
        sha256: "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
    },
    RemoteModel {
        id: "large-v3-turbo",
        name: "Large v3 Turbo",
        filename: "ggml-large-v3-turbo.bin",
        size: 1_624_555_275,
        description: "最新 Large モデル / 約 1.6 GB",
        sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    },
];

fn model_url(model: &RemoteModel) -> String {
    format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/{}/{}",
        HF_REVISION, model.filename
    )
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Verifies size and SHA-256 of a file on disk against the expected values.
#[allow(dead_code)]
pub fn verify_model_file(
    path: &Path,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| format!("Failed to open model: {}", e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total: u64 = 0;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("Failed to read model: {}", e))?;
        if n == 0 {
            break;
        }
        total += n as u64;
        hasher.update(&buf[..n]);
    }
    check_integrity(
        total,
        &to_hex(&hasher.finalize()),
        expected_size,
        expected_sha256,
    )
}

fn check_integrity(
    actual_size: u64,
    actual_sha256: &str,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<(), String> {
    if actual_size != expected_size {
        return Err(format!(
            "Model size mismatch: expected {} bytes, got {} bytes",
            expected_size, actual_size
        ));
    }
    if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
        return Err(format!(
            "Model checksum mismatch: expected {}, got {}",
            expected_sha256, actual_sha256
        ));
    }
    Ok(())
}

fn model_directory() -> Result<PathBuf, String> {
    if let Ok(custom_dir) = std::env::var("LOCAL_WHISPER_MODEL_DIR") {
        let trimmed = custom_dir.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }

    #[cfg(target_os = "macos")]
    {
        let home = std::env::var("HOME")
            .map_err(|_| "Failed to resolve HOME for model directory".to_string())?;
        return Ok(PathBuf::from(home).join("Library/Application Support/local-whisper/models"));
    }

    #[cfg(target_os = "windows")]
    {
        let appdata = std::env::var("APPDATA")
            .map_err(|_| "Failed to resolve APPDATA for model directory".to_string())?;
        return Ok(PathBuf::from(appdata).join("local-whisper/models"));
    }

    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        if let Ok(xdg_data_home) = std::env::var("XDG_DATA_HOME") {
            let trimmed = xdg_data_home.trim();
            if !trimmed.is_empty() {
                return Ok(PathBuf::from(trimmed).join("local-whisper/models"));
            }
        }
        let home = std::env::var("HOME")
            .map_err(|_| "Failed to resolve HOME for model directory".to_string())?;
        Ok(PathBuf::from(home).join(".local/share/local-whisper/models"))
    }
}

fn legacy_model_directory() -> Result<PathBuf, String> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .map(|p| p.join("vendor/whisper.cpp/models"))
        .ok_or_else(|| "Failed to resolve model directory".to_string())
}

fn model_directories_for_scan() -> Result<Vec<PathBuf>, String> {
    let primary = model_directory()?;
    let legacy = legacy_model_directory()?;
    if primary == legacy {
        Ok(vec![primary])
    } else {
        Ok(vec![primary, legacy])
    }
}

fn read_installed_models() -> Result<Vec<ModelInfo>, String> {
    let mut models = Vec::new();
    let mut seen_paths = std::collections::HashSet::new();

    for model_dir in model_directories_for_scan()? {
        if !model_dir.exists() {
            continue;
        }

        let entries = std::fs::read_dir(&model_dir)
            .map_err(|e| format!("Failed to read model directory: {}", e))?;

        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if let Some(filename) = path.file_name() {
                    let filename_str = filename.to_string_lossy();
                    if filename_str.starts_with("ggml-") && filename_str.ends_with(".bin") {
                        let path_str = path.to_string_lossy().to_string();
                        if seen_paths.contains(&path_str) {
                            continue;
                        }
                        if let Ok(metadata) = entry.metadata() {
                            seen_paths.insert(path_str.clone());
                            models.push(ModelInfo {
                                name: filename_str.to_string(),
                                path: path_str,
                                size: metadata.len(),
                            });
                        }
                    }
                }
            }
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

fn emit_model_install_progress(app_handle: Option<&AppHandle>, event: &ModelInstallProgressEvent) {
    if let Some(app) = app_handle {
        let _ = app.emit("model-install-progress", event);
    }
}

fn whisper_params_state() -> Arc<ParkingMutex<WhisperParams>> {
    WHISPER_PARAMS
        .get_or_init(|| Arc::new(ParkingMutex::new(WhisperParams::default())))
        .clone()
}

pub async fn scan_models_impl() -> Result<Vec<ModelInfo>, String> {
    read_installed_models()
}

pub async fn initialize_whisper_impl(model_path: String) -> Result<String, String> {
    let params_state = whisper_params_state();

    let mut ctx = WhisperContext::new(&model_path).map_err(|e| e.to_string())?;
    {
        let params_guard = params_state.lock();
        ctx.set_params(*params_guard);
    }

    WHISPER_CTX
        .get_or_init(|| Arc::new(Mutex::new(None)))
        .lock()
        .unwrap()
        .replace(ctx);

    Ok("Whisper initialized successfully".to_string())
}

pub async fn get_whisper_params_impl() -> Result<WhisperParamsConfig, String> {
    let state = whisper_params_state();
    let guard = state.lock();
    Ok(WhisperParamsConfig::from(*guard))
}

pub async fn set_whisper_params_impl(config: WhisperParamsConfig) -> Result<(), String> {
    let params: WhisperParams = config.into();
    let state = whisper_params_state();
    {
        let mut guard = state.lock();
        *guard = params;
    }

    if let Some(ctx_lock) = WHISPER_CTX.get() {
        let mut ctx_guard = ctx_lock.lock().unwrap();
        if let Some(ctx) = ctx_guard.as_mut() {
            ctx.set_params(params);
        }
    }

    let new_max_samples = constants::calculate_session_max_samples(params.audio_ctx);
    if let Some(recording_state) = try_recording_state() {
        let mut state_guard = recording_state.lock();
        state_guard.session_max_samples = new_max_samples;
    }

    info!(
        "Updated Whisper params: audio_ctx {}, temperature {:.2}, max session duration {:.1}s",
        params.audio_ctx,
        params.temperature,
        new_max_samples as f32 / VAD_SAMPLE_RATE as f32
    );

    Ok(())
}

pub async fn list_remote_models_impl() -> Result<Vec<RemoteModelStatus>, String> {
    let installed = read_installed_models()?;
    let mut statuses = Vec::new();

    for remote in REMOTE_MODELS {
        let installed_entry = installed.iter().find(|m| {
            Path::new(&m.path)
                .file_name()
                .map(|n| n == remote.filename)
                .unwrap_or(false)
        });

        statuses.push(RemoteModelStatus {
            id: remote.id.to_string(),
            name: remote.name.to_string(),
            filename: remote.filename.to_string(),
            size: remote.size,
            description: remote.description.to_string(),
            installed: installed_entry.is_some(),
            path: installed_entry.map(|m| m.path.clone()),
        });
    }

    Ok(statuses)
}

pub async fn install_model_impl(model_id: String) -> Result<ModelInfo, String> {
    let app_handle = APP_HANDLE.get();
    let model = REMOTE_MODELS
        .iter()
        .find(|m| m.id == model_id.as_str())
        .ok_or_else(|| "Unknown model id".to_string())?;

    // Held until this function returns; must be taken before the existence check and the
    // temp-file handling below so they cannot race with another install of the same model.
    let _install_guard = InstallGuard::acquire(model.id)
        .ok_or_else(|| format!("Model {} is already being installed", model.id))?;

    let total_bytes = model.size;
    emit_model_install_progress(
        app_handle,
        &ModelInstallProgressEvent {
            model_id: model_id.clone(),
            filename: model.filename.to_string(),
            downloaded_bytes: 0,
            total_bytes,
            percent: 0.0,
            status: "downloading".to_string(),
            message: None,
        },
    );

    let dir = model_directory()?;
    if !dir.exists() {
        std::fs::create_dir_all(&dir).map_err(|e| {
            let message = format!("Failed to create model dir {}: {}", dir.display(), e);
            emit_model_install_progress(
                app_handle,
                &ModelInstallProgressEvent {
                    model_id: model_id.clone(),
                    filename: model.filename.to_string(),
                    downloaded_bytes: 0,
                    total_bytes,
                    percent: 0.0,
                    status: "error".to_string(),
                    message: Some(message.clone()),
                },
            );
            message
        })?;
    }

    let target_path = dir.join(model.filename);
    if target_path.exists() {
        let metadata = std::fs::metadata(&target_path)
            .map_err(|e| format!("Failed to read existing model metadata: {}", e))?;
        emit_model_install_progress(
            app_handle,
            &ModelInstallProgressEvent {
                model_id: model_id.clone(),
                filename: model.filename.to_string(),
                downloaded_bytes: metadata.len(),
                total_bytes: metadata.len(),
                percent: 100.0,
                status: "completed".to_string(),
                message: None,
            },
        );
        return Ok(ModelInfo {
            name: model.filename.to_string(),
            path: target_path.to_string_lossy().to_string(),
            size: metadata.len(),
        });
    }

    use sha2::{Digest, Sha256};

    let tmp_path = target_path.with_extension("download");
    let emit_error = |downloaded: u64, total: u64, message: &str| {
        let percent = if total > 0 {
            ((downloaded as f64 / total as f64) * 100.0).min(100.0) as f32
        } else {
            0.0
        };
        emit_model_install_progress(
            app_handle,
            &ModelInstallProgressEvent {
                model_id: model_id.clone(),
                filename: model.filename.to_string(),
                downloaded_bytes: downloaded,
                total_bytes: total,
                percent,
                status: "error".to_string(),
                message: Some(message.to_string()),
            },
        );
    };

    let client = Client::builder()
        .connect_timeout(Duration::from_secs(30))
        // Idle timeout between reads; no overall timeout because models are large.
        .read_timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| {
            let message = format!("Failed to build HTTP client: {}", e);
            emit_error(0, total_bytes, &message);
            message
        })?;
    let mut response = client.get(model_url(model)).send().await.map_err(|e| {
        let message = format!("Failed to download model: {}", e);
        emit_error(0, total_bytes, &message);
        message
    })?;

    if !response.status().is_success() {
        let message = format!("Download failed with status {}", response.status());
        emit_error(0, total_bytes, &message);
        return Err(message);
    }

    if let Some(len) = response.content_length()
        && len != total_bytes
    {
        let message = format!(
            "Unexpected Content-Length: expected {} bytes, got {} bytes",
            total_bytes, len
        );
        emit_error(0, total_bytes, &message);
        return Err(message);
    }

    // A leftover temp file can only come from an interrupted earlier run (the install guard
    // excludes concurrent ones), so remove it and create a fresh file exclusively. `create_new`
    // also refuses to follow a symlink planted at the temp path.
    if let Err(e) = fs::remove_file(&tmp_path).await
        && e.kind() != std::io::ErrorKind::NotFound
    {
        let message = format!("Failed to remove stale temp file: {}", e);
        emit_error(0, total_bytes, &message);
        return Err(message);
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp_path)
        .await
        .map_err(|e| {
            let message = format!("Failed to create temp file: {}", e);
            emit_error(0, total_bytes, &message);
            message
        })?;

    let expected_total = total_bytes.max(1);
    let mut downloaded_bytes: u64 = 0;
    let mut last_notified_bytes: u64 = 0;
    let mut hasher = Sha256::new();

    // Any failure inside this block leaves a partial temp file that is removed below.
    let download_result: Result<(), String> = async {
        loop {
            let chunk = match response.chunk().await {
                Ok(Some(chunk)) => chunk,
                Ok(None) => break,
                Err(e) => return Err(format!("Failed to read download chunk: {}", e)),
            };
            downloaded_bytes += chunk.len() as u64;
            if downloaded_bytes > total_bytes {
                return Err(format!(
                    "Received more data than expected ({} bytes)",
                    total_bytes
                ));
            }
            hasher.update(&chunk);
            file.write_all(chunk.as_ref())
                .await
                .map_err(|e| format!("Failed to write chunk: {}", e))?;

            if downloaded_bytes.saturating_sub(last_notified_bytes) >= 1_048_576
                || downloaded_bytes == expected_total
            {
                let percent =
                    ((downloaded_bytes as f64 / expected_total as f64) * 100.0).min(100.0) as f32;
                emit_model_install_progress(
                    app_handle,
                    &ModelInstallProgressEvent {
                        model_id: model_id.clone(),
                        filename: model.filename.to_string(),
                        downloaded_bytes,
                        total_bytes: expected_total,
                        percent,
                        status: "downloading".to_string(),
                        message: None,
                    },
                );
                last_notified_bytes = downloaded_bytes;
            }
        }
        file.flush()
            .await
            .map_err(|e| format!("Failed to flush download: {}", e))?;
        check_integrity(
            downloaded_bytes,
            &to_hex(&hasher.clone().finalize()),
            model.size,
            model.sha256,
        )
    }
    .await;

    drop(file);
    if let Err(message) = download_result {
        let _ = fs::remove_file(&tmp_path).await;
        emit_error(downloaded_bytes, expected_total, &message);
        return Err(message);
    }

    fs::rename(&tmp_path, &target_path).await.map_err(|e| {
        let message = format!("Failed to move downloaded model: {}", e);
        emit_error(downloaded_bytes, expected_total, &message);
        message
    })?;

    let metadata = std::fs::metadata(&target_path)
        .map_err(|e| format!("Failed to read model metadata: {}", e))?;

    emit_model_install_progress(
        app_handle,
        &ModelInstallProgressEvent {
            model_id,
            filename: model.filename.to_string(),
            downloaded_bytes: metadata.len(),
            total_bytes: metadata.len(),
            percent: 100.0,
            status: "completed".to_string(),
            message: None,
        },
    );

    Ok(ModelInfo {
        name: model.filename.to_string(),
        path: target_path.to_string_lossy().to_string(),
        size: metadata.len(),
    })
}

pub async fn delete_model_impl(model_path: String) -> Result<(), String> {
    let target_path = PathBuf::from(&model_path);
    if !target_path.exists() {
        return Ok(());
    }

    let canonical_target = std::fs::canonicalize(&target_path)
        .map_err(|e| format!("Failed to resolve target path: {}", e))?;

    let allowed_dirs: Vec<PathBuf> = model_directories_for_scan()?
        .into_iter()
        .filter_map(|dir| std::fs::canonicalize(&dir).ok())
        .collect();

    if !allowed_dirs
        .iter()
        .any(|allowed_dir| canonical_target.starts_with(allowed_dir))
    {
        return Err("Invalid model path".to_string());
    }

    fs::remove_file(canonical_target)
        .await
        .map_err(|e| format!("Failed to delete model: {}", e))
}

#[cfg(test)]
mod integrity_tests {
    use super::*;
    use sha2::{Digest, Sha256};

    // SHA-256("abc")
    const ABC_SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn hex_of_known_digest_matches() {
        assert_eq!(to_hex(&Sha256::digest(b"abc")), ABC_SHA);
    }

    #[test]
    fn integrity_accepts_match() {
        assert!(check_integrity(3, ABC_SHA, 3, ABC_SHA).is_ok());
    }

    #[test]
    fn integrity_rejects_hash_and_size_mismatch() {
        assert!(check_integrity(3, ABC_SHA, 3, &"0".repeat(64)).is_err());
        assert!(check_integrity(4, ABC_SHA, 3, ABC_SHA).is_err());
    }

    #[test]
    fn verify_file_roundtrip() {
        let path = std::env::temp_dir().join(format!("lw-verify-{}.bin", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        assert!(verify_model_file(&path, 3, ABC_SHA).is_ok());
        assert!(verify_model_file(&path, 3, &"0".repeat(64)).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
