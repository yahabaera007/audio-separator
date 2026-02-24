---
title: Audio Separator
emoji: "\U0001F3B5"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Audio Separator

YouTube URLまたは音声ファイルから、ボーカルと伴奏を分離します。
オプションで話者分離（Speaker Diarization）も利用可能です。

## 話者分離のセットアップ

1. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) のライセンスに同意
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) のライセンスに同意
3. Space の Settings > Variables and Secrets で `HF_TOKEN` を Secret として設定
