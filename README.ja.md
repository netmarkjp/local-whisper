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

依存はすべてロックファイル（`mise.lock` / `Cargo.lock` / `pnpm-lock.yaml`）でバージョンとチェックサムが固定されています。以下の手順はロックファイルを書き換えずに、その内容どおりにビルドするためのものです。

### 0. 事前確認

```bash
uname -m                   # arm64 であること（Intel Mac は mise.lock の対象外。下の「注意点」参照）
sw_vers -productVersion    # 13.3 以上を推奨
xcode-select -p            # Xcode Command Line Tools が入っていること（無ければ xcode-select --install）
cmake --version            # 無ければ brew install cmake
mise --version             # 無ければ https://mise.jdx.dev/ の手順でインストール
rustup default 2>/dev/null # 既に rustup を使っている場合、現在の既定 toolchain を控えておく
```

### 1. リポジトリを取得

```bash
git clone <repository-url>
cd local-whisper
git switch <branch>        # 必要に応じて
```

### 2. ツール（Rust / Node.js / pnpm）をインストール

`mise trust` は、このディレクトリの `mise.toml` を信頼して読み込むことを許可する操作です。実行前に `mise.toml` の内容（ツールとバージョン、`[settings]`）を確認してください。

```bash
cat mise.toml
mise trust
mise install               # mise.lock があるため、チェックサムを検証してインストールされます
eval "$(mise activate zsh)"  # bash の場合は `bash`。このシェルで mise 管理のツールを有効化
```

確認（`mise.toml` のバージョンと一致すること）:

```bash
rustc --version            # 1.100.0
cargo --version            # 1.100.0（Cargo の cooldown 設定 `.cargo/config.toml` は 1.100 以上でのみ有効）
node --version             # v26.13.1
pnpm --version             # 12.21.0
```

mise を activate しない場合は、以降のコマンドすべての前に `mise exec --` を付けてください（例: `mise exec -- pnpm install`）。activate も `mise exec` もせずに実行すると、システムに入っている別バージョンの cargo / node / pnpm が使われ、cooldown やロックの前提が崩れます。

### 3. フロントエンドの依存をインストール

```bash
cd apps/desktop
pnpm install --frozen-lockfile
```

`--frozen-lockfile` を付けると `pnpm-lock.yaml` と一致しない場合は失敗し、ロックファイルは更新されません。依存パッケージの install スクリプトは pnpm 12 の既定でブロックされます（対象は macOS 用のオプション依存 `fsevents` のみで、ブロックされても動作に影響はありません）。

### 4. 開発モードで起動

```bash
pnpm tauri dev -- --locked
```

`--` 以降は cargo に渡され、`--locked` により `Cargo.lock` を更新せずにビルドします。初回は whisper.cpp（CMake）と依存クレートのビルドで数分〜十数分かかります。

### 5. リリースビルド（任意）

```bash
pnpm tauri build -- --locked
```

成果物はリポジトリ直下の `target/release/bundle/macos/`（`.app`）と `target/release/bundle/dmg/` に出力されます。署名・公証用の環境変数を設定していないため、ローカル用の未公証ビルドになります（配布には使わないでください）。

### 6. ビルド後の確認

```bash
cd ../..                   # リポジトリ直下へ
git status                 # 変更が無いこと（ロックファイルが書き換わっていないこと）
rustup default 2>/dev/null # 手順 0 で控えた既定 toolchain から変わっていないこと
```

既定 toolchain が変わっていた場合は `rustup default <元の toolchain>` で戻してください。このリポジトリ内では mise が 1.100.0 を選ぶため、既定を戻してもビルドには影響しません。

### 注意点

- **Intel Mac**: `mise.lock` には macos-arm64 と linux-x64 のチェックサムしか記録していません。Intel Mac では `mise install` が失敗するか、チェックサム検証なしでインストールされる可能性があります。Intel Mac でビルドする場合は `mise lock --platform macos-x64` でロックを追加してから行ってください。
- **権限**: マイク、システム音声、画面録画を使うため、初回利用時に macOS の許可ダイアログが出ます。`pnpm tauri dev` で起動した場合、許可の対象は起動元のターミナルアプリになります。
- **`.env`**: アプリは起動したディレクトリの `.env` と `src-tauri/.env` を読み込み、API の接続先などを上書きします。出所の分からない `.env` を置かないでください。また `.env` は `.gitignore` に含まれていないため、コミットしないよう注意してください。
- **モデル**: 文字起こしモデルはアプリ内から Hugging Face の固定リビジョンをダウンロードし、サイズと SHA-256 を検証してから保存します。保存先は `~/Library/Application Support/local-whisper/models`（環境変数 `LOCAL_WHISPER_MODEL_DIR` で変更可）です。手動で置いたモデルファイルは検証されません。
- **リモート API**: リモートの文字起こし API を使う設定にした場合、WebSocket は `ws://`（平文）でのみ接続できます。信頼できないネットワーク越しのサーバーには接続しないでください。
- **依存を更新したい場合**: この手順では依存は更新されません。更新手順は下の「メンテナンス（依存の更新）」を参照してください。

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

## メンテナンス（依存の更新）

方針: ランタイム・ライブラリ・Actions は、公開から **3 日以上** 経過した最新安定版のみを採用します（サプライチェーン対策の cooldown）。

| 対象 | cooldown / 固定の設定箇所 |
| --- | --- |
| Rust / Node.js / pnpm | `mise.toml`（`minimum_release_age = "3d"`、バージョンは完全固定）と `mise.lock`（macos-arm64 / linux-x64 のチェックサム） |
| Cargo クレート | `.cargo/config.toml`（`registry.global-min-publish-age = "3 days"`、Cargo 1.100 以上が必要）と `Cargo.lock` |
| npm パッケージ | `apps/desktop/pnpm-workspace.yaml`（`minimumReleaseAge: 4320`、単位は分）と `pnpm-lock.yaml` |
| GitHub Actions | `.github/workflows/actions.lock`（[`github/gh-actions-lock`](https://github.com/github/gh-actions-lock) **v0.4.1** で生成） |
| Dependabot | `.github/dependabot.yml`（cargo / npm / github-actions に `cooldown.default-days: 3`） |

Actions のロックファイル更新（workflow を追加したときや `uses:` を変更したときに実行）:

```bash
gh extension install github/gh-actions-lock --pin v0.4.1   # 初回のみ
gh actions-lock                                            # actions.lock の onboard / 修復
gh actions-lock --verify                                   # 読み取り専用の検証（CI でも実行）
```

`mise.lock` の更新（`mise.toml` を編集した後）: `mise lock` を実行し、`mise lock --check` で確認します。Dependabot は `mise.toml` を扱わないため、ツールのバージョンは手動で更新してください。

## 謝辞

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Tauri](https://tauri.app/)

## ライセンス

本プロジェクトは [MIT License](./LICENSE) で提供されています。
