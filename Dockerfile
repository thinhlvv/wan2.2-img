FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
runpod \
diffusers \
accelerate \
transformers \
sentencepiece \
safetensors \
protobuf \
"huggingface_hub[cli]"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
