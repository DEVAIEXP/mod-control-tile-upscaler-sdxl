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
    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

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
