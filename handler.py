import os
import json
import time
import base64
import traceback
from io import BytesIO

import runpod
import torch
from diffusers import FluxPipeline

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/flux2-klein-9b-fp8")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = None


def log(message, **kwargs):
  payload = {"message": message, **kwargs}
  print(json.dumps(payload, ensure_ascii=False), flush=True)


def image_to_base64(image):
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  return base64.b64encode(buffered.getvalue()).decode("utf-8")


def boot_debug():
  log("worker_boot_start")
  log("env", MODEL_PATH=MODEL_PATH)
  log("torch_debug", cuda_available=torch.cuda.is_available(), device=device, dtype=str(dtype))

  if torch.cuda.is_available():
    try:
      log(
        "gpu_info",
        gpu_name=torch.cuda.get_device_name(0),
        device_count=torch.cuda.device_count(),
        cuda_version=torch.version.cuda,
      )
    except Exception as e:
      log("gpu_info_error", error=str(e))

    if os.path.exists(MODEL_PATH):
      try:
        files = os.listdir(MODEL_PATH)
        log(
          "model_path_exists",
          model_path=MODEL_PATH,
          file_count=len(files),
          sample_files=files[:20],
        )
      except Exception as e:
        log("model_path_list_error", error=str(e), model_path=MODEL_PATH)
    else:
      log("model_path_missing", model_path=MODEL_PATH)


def load_model():
  global pipe

  log("model_load_start", model_path=MODEL_PATH, device=device)

  start = time.time()
  pipe = FluxPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
  ).to(device)

  elapsed = round(time.time() - start, 2)
  log("model_load_success", seconds=elapsed)


try:
  boot_debug()
  load_model()
except Exception as e:
  log(
    "startup_error",
    error=str(e),
    traceback=traceback.format_exc(),
  )
  pipe = None


def handler(job):
  global pipe

  try:
    if pipe is None:
      raise RuntimeError("Model is not loaded. Check startup logs.")

    job_input = job.get("input", {})
    prompt = job_input.get("prompt", "A beautiful sunset over the mountains")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    steps = job_input.get("steps", 4)
    guidance_scale = job_input.get("guidance_scale", 0.0)
    seed = job_input.get("seed")

    log(
      "job_received",
      prompt=prompt,
      width=width,
      height=height,
      steps=steps,
      guidance_scale=guidance_scale,
      seed=seed,
    )

    generator = torch.Generator(device=device)
    if seed is not None:
      generator.manual_seed(seed)

    start = time.time()

    with torch.inference_mode():
      result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
      )

    image = result.images[0]
    img_str = image_to_base64(image)

    elapsed = round(time.time() - start, 2)
    log("job_success", seconds=elapsed)

    return {
      "ok": True,
      "image": img_str,
      "seed": seed,
      "device": device,
      "model_path": MODEL_PATH,
      "elapsed_seconds": elapsed,
    }

  except Exception as e:
    log(
      "job_error",
      error=str(e),
      traceback=traceback.format_exc(),
    )
    return {
      "ok": False,
      "error": str(e),
      "traceback": traceback.format_exc(),
      "device": device,
      "model_path": MODEL_PATH,
    }


runpod.serverless.start({"handler": handler})
