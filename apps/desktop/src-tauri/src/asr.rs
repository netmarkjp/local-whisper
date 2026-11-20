//! Speech recognition model management: the pinned Qwen3-ASR GGUF pair, its verified
//! download, and the loaded model shared with the transcription worker.

use asr_core::AsrModel;
use log::info;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tokio::fs;
use tokio::io::AsyncWriteExt;

use crate::APP_HANDLE;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    /// Path of the text model GGUF; its `mmproj-` sibling sits in the same directory.
    pub path: String,
    /// Combined size of the text model and the mmproj.
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

#[derive(Debug)]
pub struct ModelFile {
    pub filename: &'static str,
    pub size: u64,
    /// Expected SHA-256 (lowercase hex) of the file at `HF_REVISION`.
    pub sha256: &'static str,
}

/// A Qwen3-ASR model: the text decoder plus the audio encoder/projector (mmproj).
#[derive(Debug)]
pub struct RemoteModel {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub model: ModelFile,
    pub mmproj: ModelFile,
}

impl RemoteModel {
    fn files(&self) -> [&ModelFile; 2] {
        // mmproj first: it is smaller, so a bad network fails fast.
        [&self.mmproj, &self.model]
    }

    fn total_size(&self) -> u64 {
        self.model.size + self.mmproj.size
    }
}

/// The loaded model. Transcription clones the `Arc` so the lock is not held during inference.
static ASR_MODEL: Lazy<Mutex<Option<Arc<AsrModel>>>> = Lazy::new(|| Mutex::new(None));

pub fn current_model() -> Option<Arc<AsrModel>> {
    ASR_MODEL.lock().clone()
}

/// Model ids whose installation is currently running. Concurrent installs of the same model
/// would share the temp files, so a second request is rejected while the first is in flight.
static MODEL_INSTALLS_IN_FLIGHT: Mutex<BTreeSet<&'static str>> = Mutex::new(BTreeSet::new());

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

/// Hugging Face repository holding the official llama.cpp conversion of Qwen3-ASR.
const HF_REPO: &str = "ggml-org/Qwen3-ASR-1.7B-GGUF";

/// Pinned commit of https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF (committed 2026-04-13).
///
/// To update: fetch the new commit SHA from
/// `https://huggingface.co/api/models/ggml-org/Qwen3-ASR-1.7B-GGUF` (`sha`), then run
/// `curl -s https://huggingface.co/api/models/ggml-org/Qwen3-ASR-1.7B-GGUF/tree/<sha>` and copy
/// each file's `lfs.oid` (SHA-256) and `size` into `REMOTE_MODELS`. Update this constant and
/// all entries together; never download the files just to compute hashes from untrusted copies.
const HF_REVISION: &str = "36a678687ba7d07a74ca70ccb0e36902e005fb80";

pub const REMOTE_MODELS: &[RemoteModel] = &[RemoteModel {
    id: "qwen3-asr-1.7b-q8_0",
    name: "Qwen3-ASR 1.7B (Q8_0)",
    description: "日本語を含む多言語対応 / 約 2.5 GB",
    model: ModelFile {
        filename: "Qwen3-ASR-1.7B-Q8_0.gguf",
        size: 2_165_034_944,
        sha256: "58e22d0532d4eacaf034cfac17a6fed159f37c41390c710186783be439d1fc57",
    },
    mmproj: ModelFile {
        filename: "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
        size: 355_709_344,
        sha256: "46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d",
    },
}];

const MMPROJ_PREFIX: &str = "mmproj-";

fn file_url(file: &ModelFile) -> String {
    format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        HF_REPO, HF_REVISION, file.filename
    )
}

/// The mmproj that belongs to a text model GGUF: `mmproj-<filename>` in the same directory.
fn mmproj_path_for(model_path: &Path) -> Option<PathBuf> {
    let filename = model_path.file_name()?.to_str()?;
    Some(model_path.with_file_name(format!("{MMPROJ_PREFIX}{filename}")))
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

    let home = std::env::var("HOME")
        .map_err(|_| "Failed to resolve HOME for model directory".to_string())?;
    Ok(PathBuf::from(home).join("Library/Application Support/local-whisper/models"))
}

/// Lists text model GGUFs in the model directory that have their mmproj next to them.
fn read_installed_models() -> Result<Vec<ModelInfo>, String> {
    let model_dir = model_directory()?;
    if !model_dir.exists() {
        return Ok(Vec::new());
    }

    let entries = std::fs::read_dir(&model_dir)
        .map_err(|e| format!("Failed to read model directory: {}", e))?;

    let mut models = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !filename.ends_with(".gguf") || filename.starts_with(MMPROJ_PREFIX) {
            continue;
        }
        let Some(mmproj_path) = mmproj_path_for(&path) else {
            continue;
        };
        let (Ok(model_meta), Ok(mmproj_meta)) = (entry.metadata(), std::fs::metadata(&mmproj_path))
        else {
            continue;
        };
        models.push(ModelInfo {
            name: filename.to_string(),
            path: path.to_string_lossy().to_string(),
            size: model_meta.len() + mmproj_meta.len(),
        });
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

fn emit_model_install_progress(app_handle: Option<&AppHandle>, event: &ModelInstallProgressEvent) {
    if let Some(app) = app_handle {
        let _ = app.emit("model-install-progress", event);
    }
}

pub async fn scan_models_impl() -> Result<Vec<ModelInfo>, String> {
    read_installed_models()
}

pub async fn initialize_model_impl(model_path: String) -> Result<String, String> {
    let model_path = PathBuf::from(model_path);
    let mmproj_path = mmproj_path_for(&model_path)
        .filter(|p| p.exists())
        .ok_or_else(|| format!("mmproj not found next to {}", model_path.display()))?;

    // Loading maps ~2.5 GB and initializes Metal; keep it off the async runtime threads.
    let model = tokio::task::spawn_blocking(move || AsrModel::load(&model_path, &mmproj_path))
        .await
        .map_err(|e| format!("Model loading task failed: {}", e))?
        .map_err(|e| e.to_string())?;

    ASR_MODEL.lock().replace(Arc::new(model));
    info!("ASR model initialized");

    Ok("Model initialized successfully".to_string())
}

pub async fn list_remote_models_impl() -> Result<Vec<RemoteModelStatus>, String> {
    let installed = read_installed_models()?;
    let mut statuses = Vec::new();

    for remote in REMOTE_MODELS {
        let installed_entry = installed.iter().find(|m| m.name == remote.model.filename);

        statuses.push(RemoteModelStatus {
            id: remote.id.to_string(),
            name: remote.name.to_string(),
            filename: remote.model.filename.to_string(),
            size: remote.total_size(),
            description: remote.description.to_string(),
            installed: installed_entry.is_some(),
            path: installed_entry.map(|m| m.path.clone()),
        });
    }

    Ok(statuses)
}

/// Reports install progress for a model as a whole (all of its files combined).
struct ProgressReporter<'a> {
    app_handle: Option<&'a AppHandle>,
    model_id: &'a str,
    filename: &'a str,
    total_bytes: u64,
}

impl ProgressReporter<'_> {
    fn emit(&self, downloaded_bytes: u64, status: &str, message: Option<String>) {
        let total = self.total_bytes.max(1);
        emit_model_install_progress(
            self.app_handle,
            &ModelInstallProgressEvent {
                model_id: self.model_id.to_string(),
                filename: self.filename.to_string(),
                downloaded_bytes,
                total_bytes: self.total_bytes,
                percent: ((downloaded_bytes as f64 / total as f64) * 100.0).min(100.0) as f32,
                status: status.to_string(),
                message,
            },
        );
    }
}

/// Downloads one file to `target_path` through a temp file, verifying size and SHA-256 before
/// moving it into place. `offset` is the number of bytes of the model already downloaded, so
/// progress covers the whole model. Returns the bytes downloaded for this file.
async fn download_verified_file(
    client: &Client,
    file: &ModelFile,
    target_path: &Path,
    offset: u64,
    progress: &ProgressReporter<'_>,
) -> Result<u64, String> {
    use sha2::{Digest, Sha256};

    let mut response = client
        .get(file_url(file))
        .send()
        .await
        .map_err(|e| format!("Failed to download {}: {}", file.filename, e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Download of {} failed with status {}",
            file.filename,
            response.status()
        ));
    }

    if let Some(len) = response.content_length()
        && len != file.size
    {
        return Err(format!(
            "Unexpected Content-Length for {}: expected {} bytes, got {} bytes",
            file.filename, file.size, len
        ));
    }

    // A leftover temp file can only come from an interrupted earlier run (the install guard
    // excludes concurrent ones), so remove it and create a fresh file exclusively. `create_new`
    // also refuses to follow a symlink planted at the temp path.
    let tmp_path = target_path.with_extension("download");
    if let Err(e) = fs::remove_file(&tmp_path).await
        && e.kind() != std::io::ErrorKind::NotFound
    {
        return Err(format!("Failed to remove stale temp file: {}", e));
    }
    let mut out = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp_path)
        .await
        .map_err(|e| format!("Failed to create temp file: {}", e))?;

    let mut downloaded: u64 = 0;
    let mut last_notified: u64 = 0;
    let mut hasher = Sha256::new();

    // Any failure inside this block leaves a partial temp file that is removed below.
    let result: Result<(), String> = async {
        loop {
            let chunk = match response.chunk().await {
                Ok(Some(chunk)) => chunk,
                Ok(None) => break,
                Err(e) => return Err(format!("Failed to read download chunk: {}", e)),
            };
            downloaded += chunk.len() as u64;
            if downloaded > file.size {
                return Err(format!(
                    "Received more data than expected for {} ({} bytes)",
                    file.filename, file.size
                ));
            }
            hasher.update(&chunk);
            out.write_all(chunk.as_ref())
                .await
                .map_err(|e| format!("Failed to write chunk: {}", e))?;

            if downloaded.saturating_sub(last_notified) >= 1_048_576 || downloaded == file.size {
                progress.emit(offset + downloaded, "downloading", None);
                last_notified = downloaded;
            }
        }
        out.flush()
            .await
            .map_err(|e| format!("Failed to flush download: {}", e))?;
        check_integrity(
            downloaded,
            &to_hex(&hasher.clone().finalize()),
            file.size,
            file.sha256,
        )
        .map_err(|e| format!("{}: {}", file.filename, e))
    }
    .await;

    drop(out);
    if let Err(message) = result {
        let _ = fs::remove_file(&tmp_path).await;
        return Err(message);
    }

    fs::rename(&tmp_path, target_path)
        .await
        .map_err(|e| format!("Failed to move downloaded {}: {}", file.filename, e))?;
    Ok(downloaded)
}

pub async fn install_model_impl(model_id: String) -> Result<ModelInfo, String> {
    let app_handle = APP_HANDLE.get();
    let model = REMOTE_MODELS
        .iter()
        .find(|m| m.id == model_id.as_str())
        .ok_or_else(|| "Unknown model id".to_string())?;

    // Held until this function returns; must be taken before the existence checks and the
    // temp-file handling so they cannot race with another install of the same model.
    let _install_guard = InstallGuard::acquire(model.id)
        .ok_or_else(|| format!("Model {} is already being installed", model.id))?;

    let progress = ProgressReporter {
        app_handle,
        model_id: &model_id,
        filename: model.model.filename,
        total_bytes: model.total_size(),
    };
    progress.emit(0, "downloading", None);

    let result: Result<(), String> = async {
        let dir = model_directory()?;
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("Failed to create model dir {}: {}", dir.display(), e))?;

        let client = Client::builder()
            .connect_timeout(Duration::from_secs(30))
            // Idle timeout between reads; no overall timeout because models are large.
            .read_timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

        let mut offset = 0;
        for file in model.files() {
            let target_path = dir.join(file.filename);
            // A file already in place is kept as is (files placed manually are not verified).
            if !target_path.exists() {
                download_verified_file(&client, file, &target_path, offset, &progress).await?;
            }
            offset += file.size;
        }
        Ok(())
    }
    .await;

    if let Err(message) = result {
        progress.emit(0, "error", Some(message.clone()));
        return Err(message);
    }

    let target_path = model_directory()?.join(model.model.filename);
    let size = model.total_size();
    progress.emit(size, "completed", None);

    Ok(ModelInfo {
        name: model.model.filename.to_string(),
        path: target_path.to_string_lossy().to_string(),
        size,
    })
}

/// Deletes a text model GGUF and its mmproj. Only paths inside the model directory are allowed.
pub async fn delete_model_impl(model_path: String) -> Result<(), String> {
    let target_path = PathBuf::from(&model_path);
    if !target_path.exists() {
        return Ok(());
    }

    let canonical_target = std::fs::canonicalize(&target_path)
        .map_err(|e| format!("Failed to resolve target path: {}", e))?;
    let allowed_dir = std::fs::canonicalize(model_directory()?)
        .map_err(|e| format!("Failed to resolve model directory: {}", e))?;
    if !canonical_target.starts_with(&allowed_dir) {
        return Err("Invalid model path".to_string());
    }

    if let Some(mmproj_path) = mmproj_path_for(&canonical_target)
        && let Err(e) = fs::remove_file(&mmproj_path).await
        && e.kind() != std::io::ErrorKind::NotFound
    {
        return Err(format!("Failed to delete mmproj: {}", e));
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

    #[test]
    fn mmproj_sits_next_to_model() {
        let path = Path::new("/models/Qwen3-ASR-1.7B-Q8_0.gguf");
        assert_eq!(
            mmproj_path_for(path).unwrap(),
            Path::new("/models/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
        );
    }

    #[test]
    fn remote_model_files_match_pinned_names() {
        for model in REMOTE_MODELS {
            assert_eq!(
                model.mmproj.filename,
                format!("{MMPROJ_PREFIX}{}", model.model.filename)
            );
        }
    }
}
