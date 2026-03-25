import os
import base64
from io import BytesIO

import runpod
import torch
from diffusers import FluxPipeline

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/flux2-klein-9b-fp8")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("--- Starting FLUX Runpod Serverless Worker ---")
print("MODEL_PATH =", MODEL_PATH)
print("CUDA available =", torch.cuda.is_available())

if torch.cuda.is_available():
print("GPU =", torch.cuda.get_device_name(0))
else:
print("No CUDA GPU detected")

print("Loading model...")
pipe = FluxPipeline.from_pretrained(
MODEL_PATH,
torch_dtype=dtype,
).to(device)
print("Model loaded.")

def handler(job):
job_input = job.get("input", {})
prompt = job_input.get("prompt", "A beautiful sunset over the mountains")
width = job_input.get("width", 1024)
height = job_input.get("height", 1024)
steps = job_input.get("steps", 4)
guidance_scale = job_input.get("guidance_scale", 0.0)
seed = job_input.get("seed", None)

generator = torch.Generator(device=device)
if seed is not None:
generator.manual_seed(seed)

with torch.inference_mode():
image = pipe(
prompt=prompt,
width=width,
height=height,
num_inference_steps=steps,
guidance_scale=guidance_scale,
generator=generator,
).images[0]

buffered = BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

return {
"image": img_str,
"seed": seed,
"device": device,
"model_path": MODEL_PATH,
}

runpod.serverless.start({"handler": handler})
