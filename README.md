---
title: Audio Separator
emoji: "\U0001F3B5"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Audio Separator

YouTube URLまたは音声ファイルから、ボーカルと伴奏（BGM）を分離します。
オプションで話者分離（Speaker Diarization）も利用可能です。

## 機能

- **ボーカル / BGM 分離** — UVR-MDX-NET モデルによる高品質分離
- **話者分離** — pyannote.audio による最大6人の話者検出
- **YouTube対応** — URLを貼るだけで音声を自動取得
- **最大60分** の音声に対応

## デプロイ手順

### 1. Hugging Face Space を作成

1. [huggingface.co/new-space](https://huggingface.co/new-space) にアクセス
2. Space name を入力（例: `audio-separator`）
3. SDK は **Docker** を選択
4. Visibility は **Public** を選択
5. 「Create Space」をクリック

### 2. pyannote ライセンスに同意（話者分離を使う場合）

1. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) にアクセスし、ライセンスに同意
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) にアクセスし、ライセンスに同意

### 3. HF_TOKEN を設定

1. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) でアクセストークンを作成
2. Space の **Settings → Variables and Secrets** で `HF_TOKEN` を **Secret** として追加

### 4. コードをプッシュ

```bash
cd audio_separator_space
git remote add space https://huggingface.co/spaces/<your-username>/audio-separator
git push space main
```

### 5. 完了

ビルドが完了すると、以下の URL で公開されます:

```
https://<your-username>-audio-separator.hf.space
```
