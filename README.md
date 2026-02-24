---
title: Audio Separator
emoji: "\U0001F3B5"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# Audio Separator

YouTube URLから音声をダウンロードし、ボーカルと伴奏を分離します。
オプションで話者分離（Speaker Diarization）も利用可能です。

## 機能

- YouTube URLから音声ダウンロード（yt-dlp）
- ボーカル / 伴奏の分離（UVR-MDX-NET via audio-separator）
- 話者ダイアリゼーション（pyannote.audio）

## 話者分離のセットアップ

話者分離機能を使うには以下の設定が必要です：

1. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) のライセンスに同意
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) のライセンスに同意
3. Space の Settings > Variables and Secrets で `HF_TOKEN` を Secret として設定
