import gc
import os
import shutil
import subprocess
import tempfile
import traceback

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

MAX_DURATION_SEC = 3600  # 60 minutes
MAX_SPEAKERS = 6

# Pre-load separator model at startup
print("[INFO] Loading separation model...", flush=True)
SEPARATOR_MODEL = "UVR-MDX-NET-Voc_FT.onnx"
_separator = Separator(output_format="wav")
_separator.load_model(SEPARATOR_MODEL)
print("[INFO] Model loaded.", flush=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def safe_cleanup(path):
    """一時ディレクトリを安全に削除する。"""
    try:
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def validate_youtube_url(url):
    """YouTube URL の基本的なバリデーション。"""
    url = url.strip()
    if not url:
        return None
    valid_prefixes = (
        "https://www.youtube.com/",
        "http://www.youtube.com/",
        "https://youtube.com/",
        "http://youtube.com/",
        "https://youtu.be/",
        "http://youtu.be/",
        "https://m.youtube.com/",
        "http://m.youtube.com/",
        "https://music.youtube.com/",
    )
    if not any(url.startswith(p) for p in valid_prefixes):
        raise gr.Error(
            "有効な YouTube URL を入力してください。\n"
            "例: https://www.youtube.com/watch?v=..."
        )
    return url


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
        "socket_timeout": 30,
        "retries": 3,
        "extractor_args": {"youtube": {"player_client": ["mediaconnect"]}},
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        err_msg = str(e).lower()
        if "private" in err_msg:
            raise gr.Error("この動画は非公開です。公開動画の URL を入力してください。")
        if "age" in err_msg:
            raise gr.Error("年齢制限付き動画はダウンロードできません。")
        if "unavailable" in err_msg or "not available" in err_msg:
            raise gr.Error("この動画は利用できません。URL を確認してください。")
        if "copyright" in err_msg:
            raise gr.Error("著作権の制限によりダウンロードできません。")
        raise gr.Error(f"動画のダウンロードに失敗しました: {e}")
    except Exception as e:
        raise gr.Error(f"ダウンロード中にエラーが発生しました: {e}")

    if info is None:
        raise gr.Error("動画の情報を取得できませんでした。URL を確認してください。")

    title = info.get("title", "Unknown")
    duration = info.get("duration") or 0

    if duration > MAX_DURATION_SEC:
        raise gr.Error(
            f"動画が {MAX_DURATION_SEC // 60} 分を超えています（{duration // 60}分{duration % 60}秒）。\n"
            f"最大 {MAX_DURATION_SEC // 60} 分までの動画に対応しています。"
        )

    # Find downloaded file
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
            raise gr.Error(
                "ダウンロードしたファイルが見つかりません。"
                "別の URL で試してください。"
            )

    file_size = os.path.getsize(downloaded)
    print(f"[INFO] Downloaded: {downloaded} ({file_size} bytes)", flush=True)

    if file_size == 0:
        raise gr.Error("ダウンロードしたファイルが空です。別の URL で試してください。")

    # Convert to 16kHz mono WAV
    wav_path = os.path.join(output_dir, "input.wav")
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", downloaded, "-ar", "16000", "-ac", "1", wav_path, "-y"],
            capture_output=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        raise gr.Error("音声変換がタイムアウトしました。短い動画で試してください。")

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[:500]
        raise gr.Error(f"音声の WAV 変換に失敗しました。\n詳細: {stderr}")

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) <= 0:
        raise gr.Error("WAV 変換後のファイルが空です。別の動画で試してください。")

    print(f"[INFO] WAV: {wav_path} ({os.path.getsize(wav_path)} bytes)", flush=True)

    # Clean up original download to save disk space
    try:
        if downloaded != wav_path and os.path.exists(downloaded):
            os.remove(downloaded)
    except Exception:
        pass

    return wav_path, title, duration


def convert_to_wav(input_path, output_dir):
    """任意の音声/動画ファイルを WAV に変換する。"""
    if not os.path.exists(input_path):
        raise gr.Error("アップロードされたファイルが見つかりません。再度アップロードしてください。")

    file_size = os.path.getsize(input_path)
    if file_size == 0:
        raise gr.Error("アップロードされたファイルが空です。")

    # Check file size limit (500MB)
    if file_size > 500 * 1024 * 1024:
        raise gr.Error("ファイルサイズが 500MB を超えています。")

    wav_path = os.path.join(output_dir, "input.wav")
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", wav_path, "-y"],
            capture_output=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        raise gr.Error("音声変換がタイムアウトしました。ファイルが大きすぎる可能性があります。")

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[:500]
        raise gr.Error(
            "このファイル形式は変換できませんでした。\n"
            "対応形式: mp3, wav, flac, ogg, m4a, mp4, webm\n"
            f"詳細: {stderr}"
        )

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) <= 0:
        raise gr.Error("変換後のファイルが空です。別のファイルで試してください。")

    # Get duration via ffprobe
    duration = 0
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
            capture_output=True,
            timeout=30,
        )
        duration = int(float(probe.stdout.decode().strip()))
    except Exception:
        pass

    if duration > MAX_DURATION_SEC:
        raise gr.Error(
            f"音声が {MAX_DURATION_SEC // 60} 分を超えています（{duration // 60}分{duration % 60}秒）。"
        )

    return wav_path, duration


def separate_audio(audio_path, output_dir):
    """ボーカルと伴奏に分離する。"""
    global _separator
    _separator.output_dir = output_dir

    try:
        results = _separator.separate(audio_path)
    except MemoryError:
        gc.collect()
        raise gr.Error(
            "メモリ不足で分離に失敗しました。\n"
            "短い音声で試してください。"
        )
    except Exception as e:
        raise gr.Error(f"音声分離中にエラーが発生しました: {e}")

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

    if vocals is None and instrumental is None:
        raise gr.Error("音声分離の結果ファイルが生成されませんでした。別の音声で試してください。")

    return vocals, instrumental


def diarize_speakers(vocals_path, output_dir):
    """話者ダイアリゼーション。"""
    if not DIARIZATION_AVAILABLE:
        return []

    from pydub import AudioSegment
    from pyannote.audio import Pipeline as DiarizationPipeline

    try:
        pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
        )
    except Exception as e:
        err_msg = str(e).lower()
        if "401" in err_msg or "403" in err_msg or "unauthorized" in err_msg:
            raise gr.Error(
                "話者分離モデルへのアクセスが拒否されました。\n"
                "HF_TOKEN が正しく設定されているか、"
                "pyannote/speaker-diarization-3.1 のライセンスに同意しているか確認してください。"
            )
        raise gr.Error(f"話者分離モデルの読み込みに失敗しました: {e}")

    try:
        diarization = pipeline(vocals_path)
    except MemoryError:
        gc.collect()
        raise gr.Error(
            "話者分離中にメモリ不足になりました。\n"
            "短い音声で試すか、話者分離を無効にしてください。"
        )
    except Exception as e:
        raise gr.Error(f"話者分離中にエラーが発生しました: {e}")

    try:
        audio = AudioSegment.from_wav(vocals_path)
    except Exception as e:
        raise gr.Error(f"音声ファイルの読み込みに失敗しました: {e}")

    speakers = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.setdefault(speaker, []).append(
            (int(turn.start * 1000), int(turn.end * 1000))
        )

    if not speakers:
        return []

    output_files = []
    for i, (speaker, segments) in enumerate(sorted(speakers.items()), 1):
        if i > MAX_SPEAKERS:
            break
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

def format_duration(seconds):
    """秒数を 分:秒 形式にフォーマット。"""
    m, s = divmod(int(seconds), 60)
    return f"{m}分{s}秒"


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
        duration = 0

        if has_url:
            url = validate_youtube_url(youtube_url)
            progress(0.05, desc="YouTube からダウンロード中...")
            wav_path, title, duration = download_audio(url, work_dir)
        else:
            progress(0.05, desc="音声を変換中...")
            wav_path, duration = convert_to_wav(audio_file, work_dir)
            title = os.path.splitext(os.path.basename(audio_file))[0]

        dur_text = format_duration(duration) if duration > 0 else "不明"
        progress(0.15, desc=f"ボーカルと伴奏を分離中（{dur_text}の音声）...")
        print(f"[INFO] Starting separation: {wav_path} (duration: {duration}s)", flush=True)
        vocals_path, instrumental_path = separate_audio(wav_path, output_dir)
        print("[INFO] Separation complete.", flush=True)

        # Clean up input wav to free disk
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

        speaker_files = []
        info_parts = [f"**{title}** の処理が完了しました。"]
        if duration > 0:
            info_parts[0] += f"（{dur_text}）"

        if enable_diarization:
            if DIARIZATION_AVAILABLE:
                if vocals_path and os.path.exists(vocals_path):
                    progress(0.65, desc="話者を分離中...")
                    speaker_files = diarize_speakers(vocals_path, output_dir)
                    if speaker_files:
                        info_parts.append(
                            f"{len(speaker_files)} 人の話者を検出しました。"
                        )
                    else:
                        info_parts.append("話者を検出できませんでした（音声が短すぎるか、単一話者の可能性があります）。")
                else:
                    info_parts.append("ボーカル音声が取得できなかったため、話者分離をスキップしました。")
            else:
                info_parts.append(
                    "⚠️ 話者分離を利用するには、Space の Settings で `HF_TOKEN` を Secret として設定し、\n"
                    "[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) と "
                    "[pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) のライセンスに同意してください。"
                )

        gc.collect()
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
    except MemoryError:
        gc.collect()
        raise gr.Error(
            "メモリ不足で処理に失敗しました。\n"
            "短い音声で試してください。"
        )
    except Exception as e:
        print(f"[ERROR] {traceback.format_exc()}", flush=True)
        raise gr.Error(f"予期しないエラーが発生しました: {e}")
    finally:
        # Clean up work_dir except output files that Gradio still needs
        try:
            for f in os.listdir(work_dir):
                fpath = os.path.join(work_dir, f)
                if fpath != output_dir and os.path.isfile(fpath):
                    os.remove(fpath)
        except Exception:
            pass


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------

with gr.Blocks(title="Audio Separator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Audio Separator")
    gr.Markdown(
        "YouTube URL または音声ファイルから、**ボーカル・伴奏（BGM）の分離**と"
        "**話者ごとの分離**を行います。\n\n"
        "CPU 処理のため、長い音声ほど時間がかかります。最大60分の音声に対応。"
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
        label="話者分離を有効にする（複数人の会話を分離）",
        value=DIARIZATION_AVAILABLE,
        interactive=True,
    )

    process_btn = gr.Button("処理開始", variant="primary", size="lg")
    status_md = gr.Markdown()

    with gr.Row():
        instrumental_out = gr.Audio(label="伴奏 / BGM (Instrumental)", type="filepath")
        vocals_out = gr.Audio(label="ボーカル (Vocals)", type="filepath")

    gr.Markdown("### 話者別音声")
    with gr.Row():
        speaker_outs = []
        for i in range(MAX_SPEAKERS):
            speaker_outs.append(
                gr.Audio(label=f"話者 {i + 1}", type="filepath", visible=True)
            )

    process_btn.click(
        fn=process,
        inputs=[url_input, audio_input, enable_diarization],
        outputs=[status_md, instrumental_out, vocals_out, *speaker_outs],
    )

demo.queue(max_size=5).launch(server_name="0.0.0.0", server_port=7860)
