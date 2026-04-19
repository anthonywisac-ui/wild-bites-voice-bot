FROM python:3.11-slim

# System deps for audio (opus, srtp), video (ffmpeg), and OpenCV (libxcb, libGL, libglib)
# aiortc needs libopus + libsrtp; Pipecat's SmallWebRTC imports cv2 which needs X libs even in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libopus0 \
    libopus-dev \
    libsrtp2-dev \
    libssl-dev \
    libffi-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libswscale-dev \
    libavutil-dev \
    pkg-config \
    gcc \
    g++ \
    git \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python || true \
    && pip install --no-cache-dir opencv-python-headless

# Copy app code
COPY . .

# Railway sets $PORT; default 8000 for local
ENV PORT=8000
EXPOSE 8000

CMD ["python", "bot.py"]
