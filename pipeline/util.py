# Copyright 2025 The DEVAIEXP Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gc
import cv2
import numpy as np
import torch
from PIL import Image
from gradio.themes import Default
import gradio as gr


MAX_SEED = np.iinfo(np.int32).max
SAMPLERS = {
    "DDIM": ("DDIMScheduler", {}),
    "DDIM trailing": ("DDIMScheduler", {"timestep_spacing": "trailing"}),
    "DDPM": ("DDPMScheduler", {}),
    "DEIS": ("DEISMultistepScheduler", {}),
    "Heun": ("HeunDiscreteScheduler", {}),
    "Heun Karras": ("HeunDiscreteScheduler", {"use_karras_sigmas": True}),
    "Euler": ("EulerDiscreteScheduler", {}),
    "Euler trailing": ("EulerDiscreteScheduler", {"timestep_spacing": "trailing", "prediction_type": "sample"}),
    "Euler Ancestral": ("EulerAncestralDiscreteScheduler", {}),
    "Euler Ancestral trailing": ("EulerAncestralDiscreteScheduler", {"timestep_spacing": "trailing"}),
    "DPM++ 1S": ("DPMSolverMultistepScheduler", {"solver_order": 1}),
    "DPM++ 1S Karras": ("DPMSolverMultistepScheduler", {"solver_order": 1, "use_karras_sigmas": True}),
    "DPM++ 2S": ("DPMSolverSinglestepScheduler", {"use_karras_sigmas": False}),
    "DPM++ 2S Karras": ("DPMSolverSinglestepScheduler", {"use_karras_sigmas": True}),
    "DPM++ 2M": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": False}),
    "DPM++ 2M Karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "DPM++ 2M SDE": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": False, "algorithm_type": "sde-dpmsolver++"}),
    "DPM++ 2M SDE Karras": (
        "DPMSolverMultistepScheduler",
        {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"},
    ),
    "DPM++ 3M": ("DPMSolverMultistepScheduler", {"solver_order": 3}),
    "DPM++ 3M Karras": ("DPMSolverMultistepScheduler", {"solver_order": 3, "use_karras_sigmas": True}),
    "DPM++ SDE": ("DPMSolverSDEScheduler", {"use_karras_sigmas": False}),
    "DPM++ SDE Karras": ("DPMSolverSDEScheduler", {"use_karras_sigmas": True}),
    "DPM2": ("KDPM2DiscreteScheduler", {}),
    "DPM2 Karras": ("KDPM2DiscreteScheduler", {"use_karras_sigmas": True}),
    "DPM2 Ancestral": ("KDPM2AncestralDiscreteScheduler", {}),
    "DPM2 Ancestral Karras": ("KDPM2AncestralDiscreteScheduler", {"use_karras_sigmas": True}),
    "LMS": ("LMSDiscreteScheduler", {}),
    "LMS Karras": ("LMSDiscreteScheduler", {"use_karras_sigmas": True}),
    "UniPC": ("UniPCMultistepScheduler", {}),
    "UniPC Karras": ("UniPCMultistepScheduler", {"use_karras_sigmas": True}),
    "PNDM": ("PNDMScheduler", {}),
    "Euler EDM": ("EDMEulerScheduler", {}),
    "Euler EDM Karras": ("EDMEulerScheduler", {"use_karras_sigmas": True}),
    "DPM++ 2M EDM": (
        "EDMDPMSolverMultistepScheduler",
        {"solver_order": 2, "solver_type": "midpoint", "final_sigmas_type": "zero", "algorithm_type": "dpmsolver++"},
    ),
    "DPM++ 2M EDM Karras": (
        "EDMDPMSolverMultistepScheduler",
        {
            "use_karras_sigmas": True,
            "solver_order": 2,
            "solver_type": "midpoint",
            "final_sigmas_type": "zero",
            "algorithm_type": "dpmsolver++",
        },
    ),
    "DPM++ 2M Lu": ("DPMSolverMultistepScheduler", {"use_lu_lambdas": True}),
    "DPM++ 2M Ef": ("DPMSolverMultistepScheduler", {"euler_at_final": True}),
    "DPM++ 2M SDE Lu": ("DPMSolverMultistepScheduler", {"use_lu_lambdas": True, "algorithm_type": "sde-dpmsolver++"}),
    "DPM++ 2M SDE Ef": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++", "euler_at_final": True}),
    "LCM": ("LCMScheduler", {}),
    "LCM trailing": ("LCMScheduler", {"timestep_spacing": "trailing"}),
    "TCD": ("TCDScheduler", {}),
    "TCD trailing": ("TCDScheduler", {"timestep_spacing": "trailing"}),
}

class Platinum(Default): 
    def __init__(
        self,                
    ):
        super().__init__(
            font = (
                gr.themes.GoogleFont("Karla"), 'Segoe UI Emoji', 'Public Sans', 'system-ui', 'sans-serif'
            )
        )
        self.name = "Diffusers"
        super().set(                  
            block_border_width='1px',
            block_border_width_dark='1px',            
            block_info_text_size='13px',
            block_info_text_weight='450',
            block_info_text_color='#474a50',
            block_label_background_fill='*background_fill_secondary',
            block_label_text_color='*neutral_700',
            block_title_text_color='black',
            block_title_text_weight='600',
            block_background_fill='#fcfcfc',
            body_background_fill='*background_fill_secondary',
            body_text_color='black',
            background_fill_secondary='#f8f8f8',
            border_color_accent='*primary_50',
            border_color_primary='#ededed',
            color_accent='#7367f0',
            color_accent_soft='#fcfcfc',            
            panel_background_fill='#fcfcfc',
            section_header_text_weight='600',
            checkbox_background_color='*background_fill_secondary',
            input_background_fill='white',        
            input_placeholder_color='*neutral_300',
            loader_color = '#7367f0',        
            slider_color='#7367f0',
            table_odd_background_fill='*neutral_100',
            button_small_radius='*radius_sm',
            button_primary_background_fill='linear-gradient(to bottom right, #7367f0, #9c93f4)',            
            button_primary_background_fill_hover='linear-gradient(to bottom right, #9c93f4, #9c93f4)',
            button_primary_background_fill_hover_dark='linear-gradient(to bottom right, #5e50ee, #5e50ee)',
            button_cancel_background_fill='linear-gradient(to bottom right, #fc0379, #ff88ac)',
            button_cancel_background_fill_dark='linear-gradient(to bottom right, #dc2626, #b91c1c)',
            button_cancel_background_fill_hover='linear-gradient(to bottom right, #f592c9, #f592c9)',
            button_cancel_background_fill_hover_dark='linear-gradient(to bottom right, #dc2626, #dc2626)',
            button_primary_border_color='#5949ed',
            button_primary_text_color='white',            
            button_cancel_text_color='white',
            button_cancel_text_color_dark='#dc2626',
            button_cancel_border_color='#f04668',
            button_cancel_border_color_dark='#dc2626',
            button_cancel_border_color_hover='#fe6565',
            button_cancel_border_color_hover_dark='#dc2626',
            form_gap_width='1px',
            layout_gap='5px'
        )


def select_scheduler(pipe, selected_sampler):
    import diffusers

    scheduler_class_name, add_kwargs = SAMPLERS[selected_sampler]
    config = pipe.scheduler.config
    scheduler = getattr(diffusers, scheduler_class_name)
    if selected_sampler in ("LCM", "LCM trailing"):
        config = {
            x: config[x] for x in config if x not in ("skip_prk_steps", "interpolation_type", "use_karras_sigmas")
        }
    elif selected_sampler in ("TCD", "TCD trailing"):
        config = {x: config[x] for x in config if x not in ("skip_prk_steps")}

    return scheduler.from_config(config, **add_kwargs)

# This function was copied and adapted from https://huggingface.co/spaces/gokaygokay/TileUpscalerV2, licensed under Apache 2.0.
def progressive_upscale(input_image, target_resolution, steps=3):
    """
    Progressively upscales an image to the target resolution in multiple steps.

    Args:
        input_image (PIL.Image.Image): The input image to be upscaled.
        target_resolution (int): The target resolution (width or height) in pixels.
        steps (int, optional): The number of upscaling steps. Defaults to 3.

    Returns:
        PIL.Image.Image: The upscaled image at the target resolution.
    """
    current_image = input_image.convert("RGB")
    current_size = max(current_image.size)

    # Upscale in multiple steps
    for _ in range(steps):
        if current_size >= target_resolution:
            break
        scale_factor = min(2, target_resolution / current_size)
        new_size = (int(current_image.width * scale_factor), int(current_image.height * scale_factor))
        current_image = current_image.resize(new_size, Image.LANCZOS)
        current_size = max(current_image.size)

    # Final resize to exact target resolution
    if current_size != target_resolution:
        aspect_ratio = current_image.width / current_image.height
        if current_image.width > current_image.height:
            new_size = (target_resolution, int(target_resolution / aspect_ratio))
        else:
            new_size = (int(target_resolution * aspect_ratio), target_resolution)
        current_image = current_image.resize(new_size, Image.LANCZOS)

    return current_image


# This function was copied and adapted from https://huggingface.co/spaces/gokaygokay/TileUpscalerV2, licensed under Apache 2.0.
def create_hdr_effect(original_image, hdr):
    """
    Applies an HDR (High Dynamic Range) effect to an image based on the specified intensity.

    Args:
        original_image (PIL.Image.Image): The original image to which the HDR effect will be applied.
        hdr (float): The intensity of the HDR effect, ranging from 0 (no effect) to 1 (maximum effect).

    Returns:
        PIL.Image.Image: The image with the HDR effect applied.
    """
    if hdr == 0:
        return original_image  # No effect applied if hdr is 0

    # Convert the PIL image to a NumPy array in BGR format (OpenCV format)
    cv_original = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    # Define scaling factors for creating multiple exposures
    factors = [
        1.0 - 0.9 * hdr,
        1.0 - 0.7 * hdr,
        1.0 - 0.45 * hdr,
        1.0 - 0.25 * hdr,
        1.0,
        1.0 + 0.2 * hdr,
        1.0 + 0.4 * hdr,
        1.0 + 0.6 * hdr,
        1.0 + 0.8 * hdr,
    ]

    # Generate multiple exposure images by scaling the original image
    images = [cv2.convertScaleAbs(cv_original, alpha=factor) for factor in factors]

    # Merge the images using the Mertens algorithm to create an HDR effect
    merge_mertens = cv2.createMergeMertens()
    hdr_image = merge_mertens.process(images)

    # Convert the HDR image to 8-bit format (0-255 range)
    hdr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype("uint8")

    torch_gc()
    
    # Convert the image back to RGB format and return as a PIL image
    return Image.fromarray(cv2.cvtColor(hdr_image_8bit, cv2.COLOR_BGR2RGB))


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    gc.collect()


def quantize_8bit(unet):
    if unet is None:
        return

    from peft.tuners.tuners_utils import BaseTunerLayer

    dtype = unet.dtype
    unet.to(torch.float8_e4m3fn)
    for module in unet.modules():  # revert lora modules to prevent errors with fp8
        if isinstance(module, BaseTunerLayer):
            module.to(dtype)

    if hasattr(unet, "encoder_hid_proj"):  # revert ip adapter modules to prevent errors with fp8
        if unet.encoder_hid_proj is not None:
            for module in unet.encoder_hid_proj.modules():
                module.to(dtype)
    torch_gc()
