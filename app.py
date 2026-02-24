import os
import tempfile

import gradio as gr
import yt_dlp
from audio_separator.separator import Separator

# --- Speaker diarization (optional: requires HF_TOKEN) ---
DIARIZATION_AVAILABLE = False
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if HF_TOKEN:
    try:
        from pyannote.audio import Pipeline as DiarizationPipeline  # noqa: F401
        from pydub import AudioSegment  # noqa: F401

        DIARIZATION_AVAILABLE = True
    except ImportError:
        pass

MAX_DURATION_SEC = 600  # 10 minutes limit
MAX_SPEAKERS = 4


# ------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------


def download_audio(url: str, output_dir: str) -> tuple[str, str]:
    """YouTube URL から音声を WAV でダウンロードする。"""
    output_template = os.path.join(output_dir, "input.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "match_filter": yt_dlp.utils.match_filter_func(
            f"duration <= {MAX_DURATION_SEC}"
        ),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise gr.Error(
                f"動画が {MAX_DURATION_SEC // 60} 分を超えています。短い動画で試してください。"
            )
        title = info.get("title", "Unknown")

    wav_path = os.path.join(output_dir, "input.wav")
    if not os.path.exists(wav_path):
        raise FileNotFoundError("ダウンロードした音声ファイルが見つかりません。")
    return wav_path, title


def separate_audio(audio_path: str, output_dir: str) -> tuple[str | None, str | None]:
    """ボーカルと伴奏に分離する。"""
    separator = Separator(output_dir=output_dir, output_format="wav")
    separator.load_model("UVR-MDX-NET-Voc_FT.onnx")
    results = separator.separate(audio_path)

    vocals = None
    instrumental = None
    for path in results:
        name = os.path.basename(path).lower()
        if "vocal" in name:
            vocals = path
        elif "instrumental" in name or "no_vocal" in name:
            instrumental = path

    # Fallback: audio-separator typically outputs [instrumental, vocals]
    if vocals is None and instrumental is None and len(results) >= 2:
        instrumental, vocals = results[0], results[1]

    return vocals, instrumental


def diarize_speakers(vocals_path: str, output_dir: str) -> list[str]:
    """話者ダイアリゼーションを実行し、話者ごとの音声ファイルを返す。"""
    if not DIARIZATION_AVAILABLE:
        return []

    from pydub import AudioSegment

    from pyannote.audio import Pipeline as DiarizationPipeline

    pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )
    diarization = pipeline(vocals_path)

    audio = AudioSegment.from_wav(vocals_path)

    speakers: dict[str, list[tuple[int, int]]] = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.setdefault(speaker, []).append(
            (int(turn.start * 1000), int(turn.end * 1000))
        )

    output_files = []
    for i, (speaker, segments) in enumerate(sorted(speakers.items()), 1):
        combined = AudioSegment.silent(duration=0)
        for start_ms, end_ms in segments:
            combined += audio[start_ms:end_ms]

        out_path = os.path.join(output_dir, f"speaker_{i}.wav")
        combined.export(out_path, format="wav")
        output_files.append(out_path)

    return output_files


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------


def process(url: str, enable_diarization: bool, progress=gr.Progress()):
    """YouTube URL を受け取り、分離処理を行う。"""
    if not url or not url.strip():
        raise gr.Error("YouTube の URL を入力してください。")

    work_dir = tempfile.mkdtemp(prefix="audio_sep_")
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Download
    progress(0.1, desc="YouTube からダウンロード中...")
    audio_path, title = download_audio(url, work_dir)

    # Step 2: Vocal / Instrumental separation
    progress(0.3, desc="ボーカルと伴奏を分離中...")
    vocals_path, instrumental_path = separate_audio(audio_path, output_dir)

    # Step 3: Speaker diarization (optional)
    speaker_files: list[str] = []
    info_parts = [f"**{title}** の処理が完了しました。"]

    if enable_diarization:
        if DIARIZATION_AVAILABLE:
            progress(0.6, desc="話者を分離中...")
            speaker_files = diarize_speakers(vocals_path, output_dir)
            if speaker_files:
                info_parts.append(f"{len(speaker_files)} 人の話者を検出しました。")
            else:
                info_parts.append("話者を検出できませんでした。")
        else:
            info_parts.append(
                "話者分離を利用するには、Space の Secrets に `HF_TOKEN` を設定し、"
                "[pyannote/speaker-diarization-3.1]"
                "(https://huggingface.co/pyannote/speaker-diarization-3.1) "
                "のライセンスに同意してください。"
            )

    progress(1.0, desc="完了!")

    # Pad speaker_files to exactly MAX_SPEAKERS slots
    padded = speaker_files[:MAX_SPEAKERS]
    padded += [None] * (MAX_SPEAKERS - len(padded))

    return (
        "\n\n".join(info_parts),
        instrumental_path,
        vocals_path,
        *padded,
    )


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------

with gr.Blocks(title="Audio Separator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Audio Separator")
    gr.Markdown(
        "YouTube の URL を入力して、**ボーカル・伴奏の分離**と"
        "**話者ごとの分離**を行います。\n\n"
        "CPU 処理のため、1 曲あたり数分かかる場合があります。"
    )

    with gr.Row():
        url_input = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            scale=4,
        )
        enable_diarization = gr.Checkbox(
            label="話者分離",
            value=DIARIZATION_AVAILABLE,
            interactive=True,
            scale=1,
        )

    process_btn = gr.Button("処理開始", variant="primary", size="lg")
    status_md = gr.Markdown()

    with gr.Row():
        instrumental_out = gr.Audio(label="伴奏 (Instrumental)", type="filepath")
        vocals_out = gr.Audio(label="ボーカル (Vocals)", type="filepath")

    with gr.Row():
        speaker_outs = []
        for i in range(MAX_SPEAKERS):
            speaker_outs.append(
                gr.Audio(label=f"話者 {i + 1}", type="filepath")
            )

    process_btn.click(
        fn=process,
        inputs=[url_input, enable_diarization],
        outputs=[status_md, instrumental_out, vocals_out, *speaker_outs],
    )

demo.launch(ssr=False)
