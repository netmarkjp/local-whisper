# Local Whisper


[![Blog](https://img.shields.io/badge/blog-neosophie.com-blue)](https://neosophie.com/en/blog/20260218-local-whisper)
[![Website](https://img.shields.io/badge/website-neosophie.com-green)](https://neosophie.com)

Apple Silicon (M1/M2/M3) 向けに最適化した、完全ローカル動作の高精度・低遅延文字起こしアプリです。
Web 会議や YouTube などのシステム音声も、マイク入力も、そのままテキスト化できます。

https://github.com/user-attachments/assets/f6268cb3-1534-42b4-9070-f0ae29a6dcf0

## なにができる？

- 🔒 **完全ローカル** – ネットワークアクセス不要。機密音声も安心。
- ⚡ **低遅延** – whisper.cpp ベースの最適化で、Apple Silicon の CPU/GPU を無駄なく活用。
- 🎧 **音声ソースを選べる** – マイク入力とシステム音声入力をワンタップで切り替え。
- 🎥 **録画もできる** – Web 会議や配信をキャプチャして、そのまま文字起こしに活用。
- 🗣️ **日本語に強い** – 長時間の会議記録や動画も安定して文字起こし。
- 🪄 **モデルを UI から選択** – base / small / medium / large v3 turbo を切り替えて精度と速度を調整。

## 必要環境

- Apple Silicon 搭載 Mac（macOS 13.3 以上推奨。UI の Tailwind CSS v4 が Safari 16.4 以上 / WKWebView を必要とするため）
- [mise](https://mise.jdx.dev/)（`mise.toml` に固定された Rust / Node.js / pnpm をインストールします）
- Xcode Command Line Tools（`xcode-select --install`）: Swift ブリッジ用の `swiftc` / `xcrun`、および `whisper-rs-sys` のビルドに必要な clang / libclang を含みます
- [CMake](https://cmake.org/)（`whisper-rs-sys` 経由の whisper.cpp ビルド用。例: `brew install cmake`）

## セットアップ

1. **リポジトリを取得**

   ```bash
   git clone <repository-url>
   cd local-whisper
   # 任意: whisper-rs-sys が whisper.cpp を同梱するためビルドには不要です。
   # crates/asr-core/build.rs が参照する vendor/whisper.cpp 配下の Metal リソースを取得したい場合のみ実行します。
   git submodule update --init --recursive
   ```

2. **依存関係をインストール**

   ```bash
   mise trust && mise install   # Rust / Node.js / pnpm
   eval "$(mise activate zsh)"  # bash の場合は `bash`。このシェルで mise 管理のツールを有効化します
   cd apps/desktop
   pnpm install
   ```

3. **開発モードで起動**
   ```bash
   pnpm tauri dev
   ```
   mise を activate しない場合は、コマンドの前に `mise exec --` を付けてください（例: `mise exec -- pnpm install`、`mise exec -- pnpm tauri dev`）。
   ビルド済みバイナリが欲しい場合は `pnpm tauri build` を利用してください。

## モデル選択のヒント

| モデル         | 特徴                   | 想定用途                          |
| -------------- | ---------------------- | --------------------------------- |
| base           | バランス型。初期設定。 | 日常的な会議や動画視聴            |
| small          | base より高精度        | 長時間の会議メモ                  |
| medium         | さらに高精度           | 医療/法律など誤差を減らしたい場面 |
| large v3 turbo | 最高精度ながら高速     | 字幕生成やアーカイブ用途          |

モデルはアプリ内のプルダウンから即時切り替えできます。
精度重視なら `medium / large v3 turbo`、レスポンス優先なら `base` を選ぶのがおすすめです。

## 使い方

1. `pnpm tauri dev` でアプリを起動。
2. 入力ソースを **マイク** か **システム音声** から選択。
3. [開始] でリアルタイム文字起こしを開始。必要に応じて [停止]。
4. テキストはその場でコピーしたり、ログとして保存できます。

## トラブルシューティング

- **音声が取得できない**: macOS の「システム設定 > プライバシーとセキュリティ > マイク」でアプリの権限を許可してください。
- **パフォーマンスが出ない**: 一時的に別モデルへ切り替え、または他の重いアプリを終了して CPU/GPU 負荷を下げてください。

## 謝辞

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Tauri](https://tauri.app/)

## ライセンス

本プロジェクトは [MIT License](./LICENSE) で提供されています。
