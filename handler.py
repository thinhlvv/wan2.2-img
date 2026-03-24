import runpod
import torch
from diffusers import FluxPipeline
import base64
from io import BytesIO
import os
import intel_extension_for_pytorch as ipex # Dòng quan trọng

# 1. Khởi tạo model (Chỉ chạy 1 lần khi Pod bật)
device = "cuda"
MODEL_PATH = "/workspace/flux2-klein-9b-fp8" 

print("--- Đang khởi động FLUX API ---")
# Load model bản nén FP8 để tiết kiệm VRAM và chạy nhanh
pipe = FluxPipeline.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16 # Hoặc bfloat16 tùy GPU
).to(device)

def handler(job):
    # 2. Nhận dữ liệu từ API Call
    job_input = job['input']
    prompt = job_input.get("prompt", "A beautiful sunset over the mountains")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    steps = job_input.get("steps", 4) # Bản Klein chỉ cần 4-8 steps
    seed = job_input.get("seed", None)

    generator = torch.Generator(device=device)
    if seed:
        generator.manual_seed(seed)

    # 3. Sinh ảnh
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0, # Flux Klein thường để 0.0 hoặc rất thấp
            generator=generator
        ).images[0]

    # 4. Mã hóa ảnh sang Base64 để gửi về qua API
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image": img_str}

# Khởi chạy Worker
runpod.serverless.start({"handler": handler})
