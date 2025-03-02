<h1 align="center">MoD ControlNet Tile Upscaler for SDXL🤗</h1>
<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; overflow:hidden;">
    <span>This project implements the <a href="https://arxiv.org/pdf/2408.06072">📜 MoD (Mixture-of-Diffusers)</a> tiled diffusion technique and combines it with SDXL's ControlNet Tile process.</span>
    <span>🚀 <b>Controlnet Union Power!</b> Check out the model: <a href="https://huggingface.co/xinsir/controlnet-union-sdxl-1.0">Controlnet Union</a></span>
    <span>🎨 <b>RealVisXL V5.0 for Stunning Visuals!</b> Explore it here: <a href="https://huggingface.co/SG161222/RealVisXL_V5.0">RealVisXL</a></span>
</div>


If you like the project, please give me a star! ⭐

[![GitHub](https://img.shields.io/github/stars/DEVAIEXP/mod-control-tile-upscaler-sdxl?style=socia)](https://github.com/DEVAIEXP/mod-control-tile-upscaler-sdxl)
<a href='https://huggingface.co/spaces/elismasilva/mod-control-tile-upscaler-sdxl'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a><br>
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/S6S71ACXMR)

<div style="text-align: center;">
  <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/mod_app.png" width="1536">
</div>

## About
The **MoD ControlNet Tile Upscale SDXL** is an advanced pipeline that leverages **ControlNet Tile** and **Mixture-of-Diffusers techniques**, integrating tile diffusion directly into the latent space denoising process. Designed to overcome the limitations of conventional pixel-space tile processing, this pipeline delivers **Super Resolution (SR)** upscaling for **higher-quality images, reduced processing time**, and **greater adaptability**.

By processing tiles in the latent space, the pipeline ensures smoother transitions, eliminates visible seams, and optimizes resource usage. Additionally, it supports **Unet quantization in float8**, significantly reducing GPU memory consumption while maintaining high performance. This makes it ideal for high-resolution image generation and upscaling tasks, even on hardware with limited resources. Whether you're working with large-scale images or complex visual data, the MoD ControlNet Tile Upscale SDXL provides a robust and efficient solution for achieving superior results.

### Key Features
* **Latent Space Processing:** Tiles are processed directly in the latent space, improving efficiency and quality.

* **Seamless Transitions:** Advanced weighting methods (e.g., Gaussian or Cosine) ensure smooth blending between tiles.

* **Dynamic Overlap:** Adaptive overlap calculations guarantee complete image coverage, even at high resolutions.

* **Unet Quantization in float8:** Reduces GPU memory consumption without compromising performance.

* **Scalability:** Designed to handle large-scale images (e.g., 8192x6144) with ease.

* **Flexibility:** Compatible with various resolutions and aspect ratios, maintaining consistency across different use cases.

### Why Choose MoD ControlNet Tile Upscale SDXL?
* **Higher Quality:** Latent space processing eliminates artifacts and improves detail preservation.

* **Faster Execution:** Parallel tile processing reduces overall computation time.

* **Reduced Memory Usage:** Unet quantization in float8 minimizes GPU memory requirements.

* **Adaptability:** Works seamlessly across different resolutions and tile sizes.

* **Ease of Use:** Intuitive integration with existing workflows and pipelines.

## Method Comparison

To demonstrate the advantages of **MoD ControlNet Tile Upscale SDXL**, this section presents a visual comparison with alternative upscaling methods: **ControlNet Tile** and **Lanczos Upscaler**. We use a **Ground Truth** image (original high-resolution image) as a reference to evaluate the quality and fidelity of each upscaling method.

### Common Issues with Conventional ControlNet Tile Upscaling

While ControlNet Tile is a useful technique for upscaling, especially when combined with ControlNets, the conventional pixel-space approach can suffer from several limitations that impact the final image quality.  This section visually demonstrates some of these common issues.

The images below showcase common artifacts often observed in images upscaled using conventional ControlNet Tile methods:

| <a href="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_control_tile_borders_2.png">ControlNet Tile - Visible Seams</a> | <a href="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_control_tile_borders.png">ControlNet Tile - Tile Inconsistency</a> |  <a href="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_upscaled_tile_borders_2.png">Ours - Without Seams</a> | <a href="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_upscaled_tile_borders.png">Ours - Without Inconsistency</a> 
|---|---|---|---|
| <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_control_tile_borders_2.png" width="400"> | <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_control_tile_borders.png" width="400"> |  <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_upscaled_tile_borders_2.png" width="400"> | <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_upscaled_tile_borders.png" width="400"> |

*   **ControlNet Tile - Visible Seams:** This example demonstrates the issue of **visible seams or tile boundaries** that can occur when tiles are processed and stitched together in pixel space. Observe the image, and focus on the areas **highlighted by the red dots. These red dots pinpoint visible seams**, which are evident as distinct lines or abrupt transitions in texture and tone, creating a noticeable grid-like artifact.
*   **ControlNet Tile - Tile Inconsistency:** This image highlights **inconsistencies between tiles**, where different tiles may exhibit slightly different styles, colors, or detail levels. Observe how the texture or color tone varies subtly across different tiles, leading to an uneven or patchwork appearance.

### Visual Comparison

Below, you can observe side-by-side the upscaling results using different methods. For a more detailed analysis, click on title of images and observing areas with fine details and textures.

| <a href="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_ground_truth.jpg">Ground Truth (Original)</a> | <a href="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_upscaled.png">Ours</a> | <a href="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_controltile.png">ControlNet Tile</a> | <a href="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_lanczos.png">Lanczos Upscaler</a> |
|---|---|---|---|
| <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_ground_truth_crop.jpg" width="300"> | <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_upscaled_crop.png" width="300"> | <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_controltile_crop.png" width="300"> | <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_lanczos_crop.png" width="300"> |

* **Ground Truth (Original):** Original high-resolution image, used as a quality and detail reference.
* **Ours:** Upscaling result using the proposed method, which integrates ControlNet Tile and Mixture-of-Diffusers in the latent space.
* **ControlNet Tile:** Upscaling result using the standard ControlNet Tile method.
* **Lanczos Upscaler:** Upscaling result using the Lanczos algorithm, a traditional interpolation method.

| Input 1024px vs Upscaled 4x (ours)</a> | Input 1024px vs Ground Truth |
|---|---|
| <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_input_4x.png" width="500"> | <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/1_ground_truth_4x.png" width="500"> |

## Examples Results

| Example 2: 1024 -> 4x |
|---|
| <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/2_slider.PNG"> |
| <a href="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/2_upscaled.png">View</a> | 

| Example 3: 512 -> 4x |
|---|
| <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/3_slider.PNG"> |
| <a href="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/3_upscaled.png">View</a> | 

| Example 4: 1024 -> 8x |
|---|
| <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/4_slider.png"> |
| <a href="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/4_upscaled.png">View</a> | 

| Example 5: 1024 -> 8x |
|---|
| <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/5_slider.PNG"> |
| <a href="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/5_upscaled.png">View</a> | 

## Installation

Use Python version 3.10.* and have the Python virtual environment installed.

Then run the following commands in the terminal:

**Clone repository:**
```bash
git clone https://github.com/DEVAIEXP/mod-control-tile-upscaler-sdxl.git
cd mod-control-tile-upscaler-sdxl
```

**Prepare environment:**
```bash
python -m venv venv
(for windows) .\venv\Scripts\activate
(for linux) source /venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade
pip install -r requirements.txt
```

## How to Run
**Gradio app:**
To launch the Gradio app on your local machine, execute the following command in your terminal:
```bash
python app.py
```

The following code👇 comes from [infer.py](infer.py). If you want to do quickly inference, please refer to the code in [infer.py](infer.py). 

````python
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

# initialize the models and pipeline
controlnet = ControlNetUnionModel.from_pretrained(
    "brad-twinkl/controlnet-union-sdxl-1.0-promax", torch_dtype=torch.float16
).to(device=device)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(device=device)

model_id = "SG161222/RealVisXL_V5.0"
pipe = StableDiffusionXLControlTilingPipeline.from_pretrained(
    model_id, controlnet=controlnet, vae=vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to(device)
````

To save VRAM, you can enable FP8 Quantization on UNET:
````python
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", variant="fp16", use_safetensors=True)
quantize_8bit(unet)
pipe.unet = unet
````

To save VRAM, you can enable CPU offloading, vae tiling and vae slicing
````python
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()
````

Set the scheduler. See **SAMPLERS** variable keys list on [util.py](pipeline/util.py#L26) file. 
````python
# Set selected scheduler
scheduler="UniPC" #<--set the key name here
pipe.scheduler = select_scheduler(pipe, scheduler)
````
....
````python
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
    image=control_image,
    control_image=image,
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
````

## Acknowledgements
- Our work is highly inspired by [Mixture-of-Diffusers](https://github.com/albarji/mixture-of-diffusers) and [ControlNetUnion](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) model. Thanks for their great works!
- We borrowed some ideias like **adaptative tile sizes**, **progressive_upscale** and **hdr effect** from [TileUpscalerV2](https://huggingface.co/spaces/gokaygokay/TileUpscalerV2). Thanks for your work!
- Thanks to [Andrew Svk](https://unsplash.com/pt-br/@andrew_svk?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) for using your image of Socotra Island in our nº 5 demo image.
- Thanks to the [HuggingFace](https://github.com/huggingface) team for their controlnet union pipeline used!
- Thanks to the [Gradio](https://github.com/gradio-app) gradio team for their support in new Sidebar component!

## Other DEVAIXP works
* [Mixture-of-Diffusers for SDXL Tiling Pipeline](https://github.com/DEVAIEXP/mixture-of-diffusers-sdxl-tiling) - SDXL Text-to-Image pipeline for image composition generation by using several diffusion processes in parallel, each configured with a specific prompt and settings, and focused on a particular region of the image.
* [Image Interrogator](https://github.com/DEVAIEXP/image-interrogator) - Tool for image captioning with support for large models like LLaVa, CogVml and others.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=DEVAIEXP/mod-control-tile-upscaler-sdxl&type=Date)](https://star-history.com/#DEVAIEXP/mod-control-tile-upscaler-sdxl&Date)

## License
This project is released under the [Apache 2.0](LICENSE).

## Contact
If you have any questions, please contact: contact@devaiexp.com