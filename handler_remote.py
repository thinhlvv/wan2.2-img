import runpod
import torch
from diffusers import Flux2KleinPipeline
import base64
from io import BytesIO
import os
from huggingface_hub import login

# 1. Cấu hình đường dẫn đến FOLDER chứa model FP16
# Ví dụ: /runpod-volume/flux2-klein-9b
# Note: use the fp8 instead
MODEL_PATH = os.getenv("MODEL_PATH", "black-forest-labs/FLUX.2-klein-9B")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- Đang nạp model FP16 từ: {MODEL_PATH} ---")

pipe = None

print(f"device: {device}")
print(f"is cuda: {torch.cuda.is_available()}")
torch.cuda.empty_cache()

def print_vram_usage():
    # Bộ nhớ hiện đang thực sự được sử dụng bởi các tensor
    allocated = torch.cuda.memory_allocated() / 1024**3
    # Bộ nhớ mà PyTorch đang "giữ chỗ" từ GPU (bao gồm cả phần chưa dùng)
    reserved = torch.cuda.memory_reserved() / 1024**3
    
    print(f"--- VRAM Usage ---")
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"------------------")

# Nạp model ở chế độ bfloat16 (tốt nhất cho Flux trên các card đời mới)
try:
    print_vram_usage()
    
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        # Nếu model của bạn là 1 file .safetensors duy nhất, hãy đổi thành:
        # use_safetensors=True
    )
    pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

    print_vram_usage()
    print("🚀 Model FP16 đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi khi nạp model: {e}")

def generateImage():
    prompt = "A high-quality portrait of a cybernetic owl"
    width = 720
    height = 1280
    steps = 4
    seed = 0
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator
            ).images[0]
    print_vram_usage()
    image.save("output_owl.png")

def handler(job):
    try:
        job_input = job['input']
        
        prompt = job_input.get("prompt", "A high-quality portrait of a cybernetic owl")
        width = job_input.get("width", 720)
        height = job_input.get("height", 1280)
        steps = job_input.get("steps", 4) 
        seed = job_input.get("seed", None)
        
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        # Sinh ảnh với chất lượng FP16 cao nhất
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator
            ).images[0]

        # Convert sang Base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image": img_str, "seed": seed}

    except Exception as e:
        return {"error": str(e)}

# runpod.serverless.start({"handler": handler})
generateImage()

