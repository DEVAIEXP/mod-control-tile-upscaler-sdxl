import torch
from diffusers import ControlNetUnionModel, AutoencoderKL, UNet2DConditionModel
import gradio as gr

from pipeline.mod_controlnet_tile_sr_sdxl import StableDiffusionXLControlNetTileSRPipeline
from pipeline.util import (
    SAMPLERS,
    create_hdr_effect,    
    progressive_upscale,
    quantize_8bit,
    select_scheduler,
    torch_gc,
)

device = "cuda"

MODELS = {"RealVisXL 5 Lightning": "SG161222/RealVisXL_V5.0_Lightning", 
          "RealVisXL 5": "SG161222/RealVisXL_V5.0"
         }

class Pipeline:
    def __init__(self):
        self.pipe = None
        self.controlnet = None
        self.vae = None
        self.last_loaded_model = None

    def load_model(self, model_id):
        if model_id != self.last_loaded_model:
            print(f"\n--- Loading model: {model_id} ---")
            if self.pipe is not None:                
                self.pipe.to("cpu")                
                del self.pipe
                self.pipe = None
                del self.controlnet
                self.controlnet = None
                del self.vae
                self.vae = None                
                torch_gc()     
            
            self.controlnet = ControlNetUnionModel.from_pretrained(
                    "brad-twinkl/controlnet-union-sdxl-1.0-promax", torch_dtype=torch.float16
                ).to(device=device)
            self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(device=device)

            self.pipe = StableDiffusionXLControlNetTileSRPipeline.from_pretrained(
                MODELS[model_id], controlnet=self.controlnet, vae=self.vae, torch_dtype=torch.float16, variant="fp16"
            ).to(device=device)

            unet = UNet2DConditionModel.from_pretrained(MODELS[model_id], subfolder="unet", variant="fp16", use_safetensors=True)
            quantize_8bit(unet)  # << Enable this if you have limited VRAM
            self.pipe.unet = unet

            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_tiling()
            self.pipe.enable_vae_slicing()
            self.last_loaded_model = model_id
            print(f"Model {model_id} loaded.")

    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)
      
# region functions
def predict(
    image,
    model_id,
    prompt,
    negative_prompt,
    resolution,
    hdr,
    num_inference_steps,
    denoising_strenght,
    controlnet_strength,
    tile_gaussian_sigma,
    scheduler,
    guidance_scale,
    max_tile_size,
    tile_weighting_method,
    progress=gr.Progress(track_tqdm=True),
):
    
    # Load model if changed
    load_model(model_id)

    # Set selected scheduler
    print(f"Using scheduler: {scheduler}...")
    pipeline.pipe.scheduler = select_scheduler(pipeline.pipe, scheduler)

    # Get current image size
    original_height = image.height
    original_width = image.width
    print(f"Current resolution: H:{original_height} x W:{original_width}")

    # Pre-upscale image for tiling
    control_image = create_hdr_effect(image, hdr)
    image = progressive_upscale(image, resolution)
    image = create_hdr_effect(image, hdr)

    # Update target height and width
    target_height = image.height
    target_width = image.width
    print(f"Target resolution: H:{target_height} x W:{target_width}")
    print(f"Applied HDR effect: {True if hdr > 0 else False}")

    # Calculate overlap size
    normal_tile_overlap, border_tile_overlap = pipeline.pipe.calculate_overlap(target_width, target_height)

    # Image generation
    print("Diffusion kicking in... almost done, coffee's on you!")
    generated_image = pipeline(
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
    
    return generated_image

def clear_result():
    return gr.update(value=None)

def load_model(model_name, on_load=False):
    global pipeline  # Declare pipeline as global
    if on_load and 'pipeline' not in globals(): # Prevent reload page
        pipeline = Pipeline()  # Create pipeline inside the function
        pipeline.load_model(model_name) # Load the initial model
    elif pipeline is not None and not on_load:
        pipeline.load_model(model_name) # Switch model      
    
def set_maximum_resolution(max_tile_size, current_value):
    max_scale = 8  # <- you can try increase it to 12x, 16x if you wish!
    maximum_value = max_tile_size * max_scale
    if current_value > maximum_value:
        return gr.update(maximum=maximum_value, value=maximum_value)
    return gr.update(maximum=maximum_value)

def select_tile_weighting_method(tile_weighting_method):
    return gr.update(visible=True if tile_weighting_method=="Gaussian" else False)

# endregion
css = """
body {    
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;    
    margin: 0;
    padding: 0;
}
.gradio-container {    
    border-radius: 15px;
    padding: 30px 40px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    margin: 40px 340px;    
}
.gradio-container h1 {    
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
}
.fillable {
    width: 100% !important;
    max-width: unset !important;
}
#examples_container {
    margin: auto;
    width: 90%;
}
#examples_row {
    justify-content: center;
}
#tips_row{    
    padding-left: 20px;
}
.sidebar {    
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}
.sidebar .toggle-button {    
    background: linear-gradient(90deg, #34d399, #10b981) !important;
    border: none;    
    padding: 12px 24px;
    text-transform: uppercase;
    font-weight: bold;
    letter-spacing: 1px;
    border-radius: 5px;
    cursor: pointer;
    transition: transform 0.2s ease-in-out;
}
.toggle-button:hover {
    transform: scale(1.05);
}
"""
title = """<h1 align="center">MoD ControlNet Tile Upscaler for SDXL🤗</h1>
           <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; overflow:hidden;">
                <span>This project implements the <a href="https://arxiv.org/pdf/2408.06072">📜 MoD (Mixture-of-Diffusers)</a> tiled diffusion technique and combines it with SDXL's ControlNet Tile process.</span>
                <span>💻 <a href="https://github.com/DEVAIEXP/mod-control-tile-upscaler-sdxl">GitHub Code</a></span>
                <span>🚀 <b>Controlnet Union Power!</b> Check out the model: <a href="https://huggingface.co/xinsir/controlnet-union-sdxl-1.0">Controlnet Union</a></span>
                <span>🎨 <b>RealVisXL V5.0 for Stunning Visuals!</b> Explore it here: <a href="https://huggingface.co/SG161222/RealVisXL_V5.0">RealVisXL</a></span>
           </div>
           """

tips = """
### Method
This project proposes an enhanced image upscaling method that leverages ControlNet Tile and Mixture-of-Diffusers techniques, integrating tile diffusion directly into the denoising process within the latent space.

Let's compare our method with conventional ControlNet Tile upscaling:

**Conventional ControlNet Tile:**
* Processes tiles in pixel space, potentially leading to edge artifacts during fusion.
* Processes each tile sequentially, increasing overall execution time (e.g., 16 tiles x 3 min = 48 min).
* Pixel space fusion using masks (e.g., Gaussian) can result in visible seams.
* Fixed or adaptively sized tiles and overlap can vary, causing inconsistencies.

**Proposed Method (MoD ControlNet Tile Upscaler):**
* Processes tiles in latent space, enabling smoother fusion and mitigating edge artifacts.
* Processes all tiles in parallel during denoising, drastically reducing execution time.
* Latent space fusion with dynamically calculated weights ensures seamless transitions between tiles.
* Tile size and overlap are dynamically adjusted based on the upscaling scale. For scales below 4x, fixed overlap maintains consistency.

"""

about = """

📧 **Contact**
<br>
If you have any questions or suggestions, feel free to send your question to <b>contact@devaiexp.com</b>.
"""

with gr.Blocks(css=css, theme=gr.themes.Ocean(), title="MoD ControlNet Tile Upscaler") as app:
    gr.Markdown(title)
    with gr.Row():
        with gr.Column(scale=3):                        
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Input Image", sources=["upload"], height=500)
                with gr.Column():
                    result = gr.Image(
                        label="Generated Image", show_label=True, format="png", interactive=False, scale=1, height=500, min_width=670
                    )
            with gr.Row():
                with gr.Accordion("Input Prompt", open=False):
                    with gr.Column():
                        prompt = gr.Textbox(
                            lines=2,
                            label="Prompt",
                            placeholder="Default prompt for image",
                            value="high-quality, noise-free edges, high quality, 4k, hd, 8k",
                        )
                    with gr.Column():
                        negative_prompt = gr.Textbox(
                            lines=2,
                            label="Negative Prompt (Optional)",
                            placeholder="e.g., blurry, low resolution, artifacts, poor details",
                            value="blurry, pixelated, noisy, low resolution, artifacts, poor details",
                        )
            with gr.Row():
                generate_button = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            with gr.Row(elem_id="tips_row"):
                gr.Markdown(tips)
    with gr.Sidebar(label="Parameters", open=True):
        with gr.Row(elem_id="parameters_row"):
            gr.Markdown("### General parameters")
            model = gr.Dropdown(
                label="Model", choices=list(MODELS.keys()), value=list(MODELS.keys())[1], interactive=False
            )
            tile_weighting_method = gr.Dropdown(
                label="Tile Weighting Method", choices=["Cosine", "Gaussian"], value="Cosine"
            )
            tile_gaussian_sigma = gr.Slider(label="Gaussian Sigma", minimum=0.05, maximum=1.0, step=0.01, value=0.3, visible=False)
            max_tile_size = gr.Dropdown(label="Max. Tile Size", choices=[1024, 1280], value=1024)
        with gr.Row():
            resolution = gr.Slider(minimum=128, maximum=8192, value=2048, step=128, label="Resolution")
            num_inference_steps = gr.Slider(minimum=2, maximum=100, value=30, step=1, label="Inference Steps")
            guidance_scale = gr.Slider(minimum=1, maximum=20, value=6, step=0.1, label="Guidance Scale")
            denoising_strength = gr.Slider(minimum=0.1, maximum=1, value=0.6, step=0.01, label="Denoising Strength")
            controlnet_strength = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.05, label="ControlNet Strength"
            )            
            hdr = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="HDR Effect")
        with gr.Row():
            scheduler = gr.Dropdown(
                label="Sampler",
                choices=list(SAMPLERS.keys()),
                value="UniPC",
            )
    with gr.Accordion(label="Example Images", open=True):
        with gr.Row(elem_id="examples_row"):
            with gr.Column(scale=12, elem_id="examples_container"):
                gr.Examples(
                    examples=[
                        [   "./examples/1.jpg",
                            "RealVisXL 5 Lightning",
                            prompt.value,
                            negative_prompt.value,
                            4096,
                            0.0,
                            25,
                            0.35,
                            1.0,
                            0.3,
                            "LCM",
                            4,
                            1024,
                            "Cosine"
                        ],
                        [   "./examples/1.jpg",
                            "RealVisXL 5",
                            prompt.value,
                            negative_prompt.value,
                            4096,
                            0.0,
                            35,
                            0.65,
                            1.0,
                            0.3,
                            "UniPC",
                            4,
                            1024,
                            "Cosine"
                        ],
                        [   "./examples/2.jpg",
                            "RealVisXL 5 Lightning",
                            prompt.value,
                            negative_prompt.value,
                            4096,
                            0.5,
                            25,
                            0.35,
                            1.0,
                            0.3,
                            "LCM",
                            4,
                            1024,
                            "Cosine"
                        ],
                        [   "./examples/2.jpg",
                            "RealVisXL 5",                       
                            prompt.value,
                            negative_prompt.value,
                            4096,
                            0.5,
                            35,
                            0.65,
                            1.0,
                            0.3,
                            "UniPC",
                            4,
                            1024,
                            "Cosine"
                        ],
                        [   "./examples/3.jpg",
                            "RealVisXL 5 Lightning",
                            prompt.value,
                            negative_prompt.value,
                            5120,
                            0.5,
                            25,
                            0.35,
                            1.0,
                            0.3,
                            "LCM",
                            4,
                            1280,
                            "Gaussian"
                        ],
                        [   "./examples/3.jpg",
                            "RealVisXL 5",
                            prompt.value,
                            negative_prompt.value,
                            5120,
                            0.5,
                            50,
                            0.65,
                            1.0,
                            0.3,
                            "UniPC",
                            4,
                            1280,
                            "Gaussian"
                        ],
                        [   "./examples/4.jpg",
                            "RealVisXL 5 Lightning",
                            prompt.value,
                            negative_prompt.value,
                            8192,
                            0.1,
                            25,
                            0.35,
                            1.0,
                            0.3,
                            "LCM",
                            4,
                            1024,
                            "Gaussian"
                        ],
                        [   "./examples/4.jpg",
                            "RealVisXL 5",
                            prompt.value,
                            negative_prompt.value,
                            8192,
                            0.1,
                            50,
                            0.5,
                            1.0,
                            0.3,
                            "UniPC",
                            4,
                            1024,
                            "Gaussian"
                        ],
                        [   "./examples/5.jpg",
                            "RealVisXL 5 Lightning",
                            prompt.value,
                            negative_prompt.value,
                            8192,
                            0.3,
                            25,
                            0.35,
                            1.0,
                            0.3,
                            "LCM",
                            4,
                            1024,
                            "Cosine"
                        ],
                        [   "./examples/5.jpg",
                            "RealVisXL 5",
                            prompt.value,
                            negative_prompt.value,
                            8192,
                            0.3,
                            50,
                            0.5,
                            1.0,
                            0.3,
                            "UniPC",
                            4,
                            1024,
                            "Cosine"
                        ]                        
                    ],
                    inputs=[
                        input_image,
                        model,
                        prompt,
                        negative_prompt,
                        resolution,
                        hdr,
                        num_inference_steps,
                        denoising_strength,
                        controlnet_strength,
                        tile_gaussian_sigma,
                        scheduler,
                        guidance_scale,
                        max_tile_size,
                        tile_weighting_method,
                    ],                    
                    fn=predict,
                    outputs=result,
                    cache_examples=False,
                )

    max_tile_size.select(fn=set_maximum_resolution, inputs=[max_tile_size, resolution], outputs=resolution)
    tile_weighting_method.change(fn=select_tile_weighting_method, inputs=tile_weighting_method, outputs=tile_gaussian_sigma)
    generate_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=predict,
        inputs=[
            input_image,
            model,
            prompt,
            negative_prompt,
            resolution,
            hdr,
            num_inference_steps,
            denoising_strength,
            controlnet_strength,
            tile_gaussian_sigma,
            scheduler,
            guidance_scale,
            max_tile_size,
            tile_weighting_method,
        ],
        outputs=result,
        show_progress="full"
    )
    gr.Markdown(about)
    app.load(fn=load_model, inputs=[model, gr.State(value=True)], outputs=None, concurrency_limit=1) # Load initial model on app load
app.launch(share=False)
