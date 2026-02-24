import os
import shutil
import subprocess
import tempfile

import gradio as gr
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

MAX_SPEAKERS = 4


# ------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------


def convert_to_wav(input_path: str, output_dir: str) -> str:
    """任意の音声/動画ファイルを WAV に変換する。"""
    wav_path = os.path.join(output_dir, "input.wav")
    subprocess.run(
        ["ffmpeg", "-i", input_path, "-ar", "44100", "-ac", "2", wav_path, "-y"],
        capture_output=True,
        check=True,
    )
    return wav_path


def separate_audio(audio_path: str, output_dir: str):
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

    speakers = {}
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


def process(audio_file, enable_diarization, progress=gr.Progress()):
    """アップロードされた音声ファイルを処理する。"""
    if audio_file is None:
        raise gr.Error("音声ファイルをアップロードしてください。")

    work_dir = tempfile.mkdtemp(prefix="audio_sep_")
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Step 1: Convert to WAV
        progress(0.1, desc="音声を変換中...")
        wav_path = convert_to_wav(audio_file, work_dir)

        # Step 2: Vocal / Instrumental separation
        progress(0.3, desc="ボーカルと伴奏を分離中...")
        vocals_path, instrumental_path = separate_audio(wav_path, output_dir)

        # Step 3: Speaker diarization (optional)
        speaker_files = []
        info_parts = ["処理が完了しました。"]

        if enable_diarization:
            if DIARIZATION_AVAILABLE:
                progress(0.6, desc="話者を分離中...")
                speaker_files = diarize_speakers(vocals_path, output_dir)
                if speaker_files:
                    info_parts.append(
                        f"{len(speaker_files)} 人の話者を検出しました。"
                    )
                else:
                    info_parts.append("話者を検出できませんでした。")
            else:
                info_parts.append(
                    "話者分離を利用するには、Space の Secrets に HF_TOKEN を設定し、"
                    "pyannote/speaker-diarization-3.1 のライセンスに同意してください。"
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

    except subprocess.CalledProcessError:
        raise gr.Error("音声ファイルの変換に失敗しました。対応形式か確認してください。")
    except Exception as e:
        raise gr.Error(f"処理中にエラーが発生しました: {e}")


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------

with gr.Blocks(title="Audio Separator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Audio Separator")
    gr.Markdown(
        "音声ファイルをアップロードして、**ボーカル・伴奏の分離**と"
        "**話者ごとの分離**を行います。\n\n"
        "CPU 処理のため、1 曲あたり数分かかる場合があります。"
    )

    with gr.Row():
        audio_input = gr.Audio(
            label="音声ファイル",
            type="filepath",
            sources=["upload"],
        )
        enable_diarization = gr.Checkbox(
            label="話者分離",
            value=DIARIZATION_AVAILABLE,
            interactive=True,
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
        inputs=[audio_input, enable_diarization],
        outputs=[status_md, instrumental_out, vocals_out, *speaker_outs],
    )

demo.launch()
