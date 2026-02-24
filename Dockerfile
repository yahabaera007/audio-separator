FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the separation model at build time
RUN python -c "from audio_separator.separator import Separator; s = Separator(); s.load_model('UVR-MDX-NET-Voc_FT.onnx')"

COPY app.py .

# Ensure tmp and app dirs are writable
RUN mkdir -p /tmp/audio_sep && \
    chown -R user:user /app /tmp/audio_sep

USER user

ENV TMPDIR=/tmp/audio_sep
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
