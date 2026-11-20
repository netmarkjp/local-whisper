# Local Whisper

[🇯🇵 日本語版 README](./README.ja.md)

[![Blog](https://img.shields.io/badge/blog-neosophie.com-blue)](https://neosophie.com/en/blog/20260218-local-whisper)
[![Website](https://img.shields.io/badge/website-neosophie.com-green)](https://neosophie.com)

Local Whisper is a high-accuracy, low-latency transcription app that runs entirely on Apple Silicon (M1/M2/M3).
It captures both microphone input and system audio (e.g., web meetings, YouTube) without sending data to the cloud.



https://github.com/user-attachments/assets/f6268cb3-1534-42b4-9070-f0ae29a6dcf0



## Highlights

- 🔒 **Fully offline** – No network access required, keeping sensitive audio on your machine.
- ⚡ **Optimized for low latency** – Runs [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) on llama.cpp with Metal acceleration on Apple Silicon.
- 🎧 **Flexible sources** – Switch between microphone input and system audio with one tap.
- 🎥 **Screen/audio capture ready** – Record meetings or streams and transcribe them immediately.
- 🗣️ **Japanese-first experience** – Tuned for long-form Japanese speech, but works for other languages too.
- 🪄 **In-app model install** – Download the speech recognition model from the interface, verified against a pinned checksum.

## Requirements

- Apple Silicon Mac (macOS 13.3+ recommended; the UI uses Tailwind CSS v4, which requires Safari 16.4+ / WKWebView)
- [mise](https://mise.jdx.dev/) (installs the pinned Rust, Node.js and pnpm from `mise.toml`)
- Xcode Command Line Tools (`xcode-select --install`): provides `swiftc` / `xcrun` for the Swift bridge, and clang / libclang for building `llama-cpp-sys-2`. Xcode itself is not needed: llama.cpp embeds its Metal shaders and compiles them at run time
- [CMake](https://cmake.org/) (for building llama.cpp via `llama-cpp-sys-2`, e.g. `brew install cmake`)

## Setup

All dependencies are pinned, with checksums, by lockfiles (`mise.lock` / `Cargo.lock` / `pnpm-lock.yaml`). The steps below build exactly what those lockfiles specify, without rewriting them.

### 0. Pre-flight checks

```bash
uname -m                   # must be arm64 (Intel Macs are not covered by mise.lock; see "Notes")
sw_vers -productVersion    # 13.3 or later recommended
xcode-select -p            # Xcode Command Line Tools installed (otherwise: xcode-select --install)
cmake --version            # otherwise: brew install cmake
mise --version             # otherwise install it from https://mise.jdx.dev/
rustup default 2>/dev/null # if you already use rustup, note your current default toolchain
```

### 1. Clone the repository

```bash
git clone <repository-url>
cd local-whisper
git switch <branch>        # if needed
```

### 2. Install the tools (Rust / Node.js / pnpm)

`mise trust` allows mise to load this directory's `mise.toml`. Review its contents (tools, versions and `[settings]`) before trusting it.

```bash
cat mise.toml
mise trust
mise install               # verifies checksums from mise.lock while installing
eval "$(mise activate zsh)"  # or `bash`; activates the mise-managed tools in this shell
```

Check that the versions match `mise.toml`:

```bash
rustc --version            # 1.100.0
cargo --version            # 1.100.0 (the Cargo cooldown in .cargo/config.toml needs Cargo 1.100+)
node --version             # v26.13.1
pnpm --version             # 12.21.0
```

If you do not activate mise, prefix every following command with `mise exec --` (e.g. `mise exec -- pnpm install`). Running without either uses whatever cargo / node / pnpm is installed system-wide, which defeats the cooldown and lockfile guarantees.

### 3. Install frontend dependencies

```bash
cd apps/desktop
pnpm install --frozen-lockfile
```

`--frozen-lockfile` fails instead of updating `pnpm-lock.yaml` if they disagree. Dependency install scripts are blocked by pnpm 12's defaults (the only one is the optional macOS dependency `fsevents`, which works without it).

### 4. Run in development mode

```bash
pnpm tauri dev -- --locked
```

Arguments after `--` are passed to cargo; `--locked` builds without updating `Cargo.lock`. The first build compiles llama.cpp (CMake) and all crates and takes several minutes or more.

### 5. Release build (optional)

```bash
pnpm tauri build -- --locked
```

Artifacts are written to `target/release/bundle/macos/` (`.app`) and `target/release/bundle/dmg/` at the repository root. Without the signing/notarization environment variables this is an un-notarized local build; do not distribute it.

### 6. Post-build checks

```bash
cd ../..                   # back to the repository root
git status                 # must be clean (no lockfile was rewritten)
rustup default 2>/dev/null # must match the default you noted in step 0
```

If your default toolchain changed, restore it with `rustup default <previous toolchain>`. Inside this repository mise still selects 1.100.0, so restoring the default does not affect the build.

### Notes

- **Intel Macs**: `mise.lock` only records checksums for macos-arm64 and linux-x64. On an Intel Mac, `mise install` may fail or install without checksum verification. Add the platform first with `mise lock --platform macos-x64`.
- **Permissions**: the app uses the microphone, system audio and screen recording, so macOS asks for permission on first use. With `pnpm tauri dev`, the permission is granted to the terminal app that launched it.
- **`.env`**: the app loads `.env` and `src-tauri/.env` from its working directory, which can override settings such as the API endpoint. Do not keep `.env` files of unknown origin there. `.env` is not in `.gitignore`, so take care not to commit it.
- **Models**: transcription models are downloaded in-app from a pinned Hugging Face revision and saved only after their size and SHA-256 are verified, to `~/Library/Application Support/local-whisper/models` (override with `LOCAL_WHISPER_MODEL_DIR`). Model files placed there manually are not verified.
- **Remote API**: if you configure a remote transcription API, the WebSocket can only connect via `ws://` (plaintext). Do not connect to servers across untrusted networks.
- **Updating dependencies**: these steps never update dependencies. See "Maintenance (updating dependencies)" below.

## Model

The app uses [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) in the official llama.cpp conversion ([ggml-org/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF), Q8_0). A model is a pair of files, about 2.5 GB in total:

| File | Role |
| --- | --- |
| `Qwen3-ASR-1.7B-Q8_0.gguf` | Text decoder |
| `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` | Audio encoder and projector |

Install it from the in-app model list. Both files must sit side by side in the model directory.

Qwen3-ASR does not return timestamps. While you speak, the app re-transcribes the current utterance every few seconds and replaces the pending text; the text is finalized when a pause is detected or the utterance reaches 30 seconds.

## How to use

1. Start the app with `pnpm tauri dev`.
2. Select the input source (**Microphone** or **System Audio**).
3. Hit **Start** to begin live transcription, and **Stop** when finished.
4. Copy results instantly or save them as logs.

## Troubleshooting

- **No audio detected**: Check macOS “System Settings → Privacy & Security → Microphone” and allow access for the app.
- **Slow performance**: Close other GPU-intensive applications, or lengthen the partial transcript interval in the streaming settings.

## Maintenance (updating dependencies)

Policy: runtimes, libraries and Actions are adopted only when the latest stable release is at least **3 days old** (supply-chain cooldown).

| Target | Where the cooldown / pin lives |
| --- | --- |
| Rust / Node.js / pnpm | `mise.toml` (`minimum_release_age = "3d"`, exact versions) and `mise.lock` (checksums for macos-arm64 and linux-x64) |
| Cargo crates | `.cargo/config.toml` (`registry.global-min-publish-age = "3 days"`, needs Cargo >= 1.100) and `Cargo.lock` |
| npm packages | `apps/desktop/pnpm-workspace.yaml` (`minimumReleaseAge: 4320`, minutes) and `pnpm-lock.yaml` |
| GitHub Actions | `.github/workflows/actions.lock`, generated by [`github/gh-actions-lock`](https://github.com/github/gh-actions-lock) **v0.4.1** |
| Dependabot | `.github/dependabot.yml` (`cooldown.default-days: 3` for cargo / npm / github-actions) |

Updating the Actions lockfile (run after adding a workflow or changing any `uses:`):

```bash
gh extension install github/gh-actions-lock --pin v0.4.1   # first time only
gh actions-lock                                            # onboard / fix actions.lock
gh actions-lock --verify                                   # read-only check (also run in CI)
```

Updating `mise.lock` (after editing `mise.toml`): run `mise lock`, and check it with `mise lock --check`. Dependabot does not manage `mise.toml`, so bump the tool versions by hand.

## Credits

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- [Tauri](https://tauri.app/)

## License

Distributed under the [MIT License](./LICENSE).

---

Built by [Neosophie](https://neosophie.com)
