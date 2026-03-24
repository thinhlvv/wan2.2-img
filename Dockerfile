FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN pip install runpod diffusers accelerate transformers sentencepiece
RUN pip install "huggingface_hub[cli]"

COPY handler.py /handler.py

# Lệnh khởi chạy
CMD [ "python", "-u", "/handler.py" ]
