FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the separation model at build time
RUN python -c "from audio_separator.separator import Separator; s = Separator(); s.load_model('UVR-MDX-NET-Voc_FT.onnx')"

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]
