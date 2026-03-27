FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cài đặt thư viện hệ thống
RUN apt-get update && apt-get install -y python3-venv libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Giữ nguyên torch từ base image; chỉ cài các thư viện tương thích với model/pipeline.
RUN python -m pip install --no-cache-dir --upgrade \
    runpod \
    diffusers \
    "transformers==4.56.1" \
    accelerate \
    sentencepiece \
    omegaconf \
    safetensors \
    protobuf

COPY handler.py /handler.py

# Đảm bảo port API của RunPod được mở (mặc định)
CMD [ "python", "/handler.py" ]
