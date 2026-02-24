import os
import subprocess
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

MAX_DURATION_SEC = 300  # 5 minutes (CPU limit)
MAX_SPEAKERS = 4

# Pre-load separator model at startup for faster processing
print("[INFO] Loading separation model...", flush=True)
SEPARATOR_MODEL = "UVR-MDX-NET-Voc_FT.onnx"
_separator = Separator(output_format="wav")
_separator.load_model(SEPARATOR_MODEL)
print("[INFO] Model loaded.", flush=True)


# ------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------


def download_audio(url, output_dir):
    """YouTube URL から音声をダウンロードし WAV に変換する。"""
    before = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()

    output_template = os.path.join(output_dir, "dl_input.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise gr.Error("動画の情報を取得できませんでした。")
        title = info.get("title", "Unknown")
        duration = info.get("duration", 0)

        if duration > MAX_DURATION_SEC:
            raise gr.Error(
                f"動画が {MAX_DURATION_SEC // 60} 分を超えています（{duration}秒）。"
                "短い動画で試してください。"
            )

        try:
            downloaded = ydl.prepare_filename(info)
        except Exception:
            downloaded = None

    if not downloaded or not os.path.exists(downloaded):
        after = set(os.listdir(output_dir))
        new_files = after - before
        if new_files:
            downloaded = os.path.join(output_dir, new_files.pop())
        else:
            files_in_dir = os.listdir(output_dir)
            raise FileNotFoundError(
                f"ダウンロードファイルが見つかりません: {files_in_dir}"
            )

    print(f"[INFO] Downloaded: {downloaded} ({os.path.getsize(downloaded)} bytes)", flush=True)

    # Convert to 16kHz mono WAV (lighter for CPU processing)
    wav_path = os.path.join(output_dir, "input.wav")
    result = subprocess.run(
        ["ffmpeg", "-i", downloaded, "-ar", "16000", "-ac", "1", wav_path, "-y"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise gr.Error(f"WAV変換に失敗: {result.stderr.decode()[:300]}")

    wav_size = os.path.getsize(wav_path) if os.path.exists(wav_path) else 0
    print(f"[INFO] WAV: {wav_path} ({wav_size} bytes)", flush=True)
    if wav_size <= 0:
        raise gr.Error("WAV変換後のファイルが空です。")

    return wav_path, title


def convert_to_wav(input_path, output_dir):
    """任意の音声/動画ファイルを WAV に変換する。"""
    wav_path = os.path.join(output_dir, "input.wav")
    subprocess.run(
        ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", wav_path, "-y"],
        capture_output=True,
        check=True,
    )
    return wav_path


def separate_audio(audio_path, output_dir):
    """ボーカルと伴奏に分離する。"""
    global _separator
    _separator.output_dir = output_dir
    results = _separator.separate(audio_path)

    print(f"[INFO] Separator results: {results}", flush=True)

    vocals = None
    instrumental = None
    for path in results:
        name = os.path.basename(path).lower()
        if "vocal" in name:
            vocals = path
        elif "instrumental" in name or "no_vocal" in name:
            instrumental = path

    if vocals is None and instrumental is None and len(results) >= 2:
        instrumental, vocals = results[0], results[1]

    for label, fpath in [("vocals", vocals), ("instrumental", instrumental)]:
        if fpath and os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"[INFO] {label}: {fpath} ({size} bytes)", flush=True)
        else:
            print(f"[WARN] {label}: not found ({fpath})", flush=True)

    return vocals, instrumental


def diarize_speakers(vocals_path, output_dir):
    """話者ダイアリゼーション。"""
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


def process(youtube_url, audio_file, enable_diarization, progress=gr.Progress()):
    """YouTube URL またはアップロード音声を処理する。"""
    has_url = youtube_url and youtube_url.strip()
    has_file = audio_file is not None

    if not has_url and not has_file:
        raise gr.Error("YouTube URL を入力するか、音声ファイルをアップロードしてください。")

    work_dir = tempfile.mkdtemp(prefix="audio_sep_")
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        if has_url:
            progress(0.1, desc="YouTube からダウンロード中...")
            wav_path, title = download_audio(youtube_url.strip(), work_dir)
        else:
            progress(0.1, desc="音声を変換中...")
            wav_path = convert_to_wav(audio_file, work_dir)
            title = os.path.splitext(os.path.basename(audio_file))[0]

        progress(0.2, desc="ボーカルと伴奏を分離中（数分かかります）...")
        print(f"[INFO] Starting separation: {wav_path}", flush=True)
        vocals_path, instrumental_path = separate_audio(wav_path, output_dir)
        print("[INFO] Separation complete.", flush=True)

        speaker_files = []
        info_parts = [f"**{title}** の処理が完了しました。"]

        if enable_diarization:
            if DIARIZATION_AVAILABLE:
                progress(0.7, desc="話者を分離中...")
                speaker_files = diarize_speakers(vocals_path, output_dir)
                if speaker_files:
                    info_parts.append(
                        f"{len(speaker_files)} 人の話者を検出しました。"
                    )
                else:
                    info_parts.append("話者を検出できませんでした。")
            else:
                info_parts.append(
                    "話者分離を利用するには、Variables に HF_TOKEN を設定し、"
                    "pyannote/speaker-diarization-3.1 のライセンスに同意してください。"
                )

        progress(1.0, desc="完了!")

        padded = speaker_files[:MAX_SPEAKERS]
        padded += [None] * (MAX_SPEAKERS - len(padded))

        return (
            "\n\n".join(info_parts),
            instrumental_path,
            vocals_path,
            *padded,
        )

    except gr.Error:
        raise
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        raise gr.Error(f"処理中にエラーが発生しました: {e}")


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------

with gr.Blocks(title="Audio Separator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Audio Separator")
    gr.Markdown(
        "YouTube URL または音声ファイルから、**ボーカル・伴奏の分離**と"
        "**話者ごとの分離**を行います。\n\n"
        "CPU 処理のため数分かかります。最大5分の音声に対応。"
    )

    with gr.Row():
        url_input = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            scale=3,
        )
        gr.Markdown("**または**")
        audio_input = gr.Audio(
            label="音声ファイルをアップロード",
            type="filepath",
            sources=["upload"],
            scale=3,
        )

    enable_diarization = gr.Checkbox(
        label="話者分離を有効にする",
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
        inputs=[url_input, audio_input, enable_diarization],
        outputs=[status_md, instrumental_out, vocals_out, *speaker_outs],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
