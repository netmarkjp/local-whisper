# Local Whisper

[🇯🇵 日本語版 README](./README.ja.md)

[![Blog](https://img.shields.io/badge/blog-neosophie.com-blue)](https://neosophie.com/en/blog/20260218-local-whisper)
[![Website](https://img.shields.io/badge/website-neosophie.com-green)](https://neosophie.com)

Local Whisper is a high-accuracy, low-latency transcription app that runs entirely on Apple Silicon (M1/M2/M3).
It captures both microphone input and system audio (e.g., web meetings, YouTube) without sending data to the cloud.



https://github.com/user-attachments/assets/f6268cb3-1534-42b4-9070-f0ae29a6dcf0



## Highlights

- 🔒 **Fully offline** – No network access required, keeping sensitive audio on your machine.
- ⚡ **Optimized for low latency** – Built on whisper.cpp to make the most of Apple Silicon’s CPU/GPU.
- 🎧 **Flexible sources** – Switch between microphone input and system audio with one tap.
- 🎥 **Screen/audio capture ready** – Record meetings or streams and transcribe them immediately.
- 🗣️ **Japanese-first experience** – Tuned for long-form Japanese speech, but works for other languages too.
- 🪄 **UI-selectable models** – Choose between base / small / medium / large v3 turbo directly from the interface.

## Requirements

- Apple Silicon Mac (macOS 13.3+ recommended; the UI uses Tailwind CSS v4, which requires Safari 16.4+ / WKWebView)
- [mise](https://mise.jdx.dev/) (installs the pinned Rust, Node.js and pnpm from `mise.toml`)
- Xcode Command Line Tools (`xcode-select --install`): provides `swiftc` / `xcrun` for the Swift bridge, and clang / libclang for building `whisper-rs-sys`
- [CMake](https://cmake.org/) (for building whisper.cpp via `whisper-rs-sys`, e.g. `brew install cmake`)

## Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd local-whisper
   # Optional: whisper-rs-sys bundles whisper.cpp, so this is not required to build.
   # It only provides the Metal resources under vendor/whisper.cpp referenced by crates/asr-core/build.rs.
   git submodule update --init --recursive
   ```

2. **Install dependencies**

   ```bash
   mise trust && mise install   # Rust / Node.js / pnpm
   eval "$(mise activate zsh)"  # or `bash`; activates the mise-managed tools in this shell
   cd apps/desktop
   pnpm install
   ```

3. **Run in development mode**
   ```bash
   pnpm tauri dev
   ```
   If you did not activate mise, prefix commands with `mise exec --`, e.g. `mise exec -- pnpm install` and `mise exec -- pnpm tauri dev`.
   Need a packaged build? Run `pnpm tauri build`.

## Model selection tips

| Model          | Characteristics                   | Suggested use case                                              |
| -------------- | --------------------------------- | --------------------------------------------------------------- |
| base           | Balanced default                  | Everyday meetings, casual videos                                |
| small          | Higher accuracy than base         | Long meeting notes                                              |
| medium         | Even higher accuracy              | Fields that require fewer transcription errors (legal, medical) |
| large v3 turbo | Highest accuracy while still fast | Subtitle generation, archival transcripts                       |

Switch models from the in-app dropdown at any time.
Pick `medium` or `large v3 turbo` when accuracy matters most; choose `base` for faster turnaround.

## How to use

1. Start the app with `pnpm tauri dev`.
2. Select the input source (**Microphone** or **System Audio**).
3. Hit **Start** to begin live transcription, and **Stop** when finished.
4. Copy results instantly or save them as logs.

## Troubleshooting

- **No audio detected**: Check macOS “System Settings → Privacy & Security → Microphone” and allow access for the app.
- **Slow performance**: Temporarily switch to a lighter model or close other CPU/GPU-intensive applications.

## Credits

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Tauri](https://tauri.app/)

## License

Distributed under the [MIT License](./LICENSE).

---

Built by [Neosophie](https://neosophie.com)
