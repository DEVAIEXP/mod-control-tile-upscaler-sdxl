import torch
from diffusers import ControlNetUnionModel, AutoencoderKL, UNet2DConditionModel
from diffusers.utils import load_image
from pipeline.mod_controlnet_tile_sr_sdxl import StableDiffusionXLControlNetTileSRPipeline, TileWeightingMethod, calculate_overlap

from pipeline.util import (
    SAMPLERS,
    create_hdr_effect,
    progressive_upscale,
    quantize_8bit,
    select_scheduler,
)

device = "cuda"

# Initialize the models and pipeline
controlnet = ControlNetUnionModel.from_pretrained(
    "brad-twinkl/controlnet-union-sdxl-1.0-promax", torch_dtype=torch.float16
).to(device=device)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(device=device)

model_id = "SG161222/RealVisXL_V5.0"
pipe = StableDiffusionXLControlNetTileSRPipeline.from_pretrained(
    model_id, controlnet=controlnet, vae=vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to(device)

unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", variant="fp16", use_safetensors=True)
quantize_8bit(unet)  # << Enable this if you have limited VRAM
pipe.unet = unet

pipe.enable_model_cpu_offload()  # << Enable this if you have limited VRAM
pipe.enable_vae_tiling() # << Enable this if you have limited VRAM
pipe.enable_vae_slicing() # << Enable this if you have limited VRAM

# Set selected scheduler
scheduler="UniPC" #<-- See samples keys list on SAMPLERS variable
pipe.scheduler = select_scheduler(pipe, scheduler)

# Load image
image = load_image("./examples/1.jpg")
original_height = image.height
original_width = image.width
print(f"Current resolution: H:{original_height} x W:{original_width}")

# Pre-upscale image for tiling
resolution = 4096
hdr = 0.5
tile_gaussian_sigma = 0.3
max_tile_size = 1024 # or 1280
control_image = create_hdr_effect(image, hdr)
image = progressive_upscale(image, resolution)
image = create_hdr_effect(image, hdr)

# Update target height and width
target_height = image.height
target_width = image.width
print(f"Target resolution: H:{target_height} x W:{target_width}")
print(f"Applied HDR effect: {True if hdr > 0 else False}")

# Calculate overlap size
normal_tile_overlap, border_tile_overlap = calculate_overlap(target_width, target_height)

# Set other params
tile_weighting_method = TileWeightingMethod.COSINE.value
guidance_scale = 4
num_inference_steps = 35
denoising_strenght = 0.65
controlnet_strength = 1.0
prompt = "high-quality, noise-free edges, high quality, 4k, hd, 8k"
negative_prompt = "blurry, pixelated, noisy, low resolution, artifacts, poor details"

# Image generation
image = pipe(
    image=image,
    control_image=control_image,
    control_mode=[6],
    controlnet_conditioning_scale=float(controlnet_strength),
    prompt=prompt,
    negative_prompt=negative_prompt,
    normal_tile_overlap=normal_tile_overlap,
    border_tile_overlap=border_tile_overlap,
    height=target_height,
    width=target_width,
    original_size=(original_width, original_height),
    target_size=(target_width, target_height),
    guidance_scale=guidance_scale,        
    strength=float(denoising_strenght),
    tile_weighting_method=tile_weighting_method,
    max_tile_size=max_tile_size,
    tile_gaussian_sigma=float(tile_gaussian_sigma),
    num_inference_steps=num_inference_steps,
)["images"][0]

image.save("result.png")

