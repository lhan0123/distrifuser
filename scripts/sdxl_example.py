import torch

import sys
from distrifuser.models.distri_clipscore import evaluate_quality
from distrifuser.pipelines import CachedSDXLPipeline
from distrifuser.utils import DistriConfig
from get_patch_maps import get_patch_map
from PIL import Image

is_profiling = "-p" in sys.argv
distri_config = DistriConfig(is_profile=is_profiling)
pipeline = CachedSDXLPipeline.from_pretrained(
    distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True,
)


pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
if is_profiling:
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    image = pipeline(
        prompt=prompt,
        generator=torch.Generator(device="cuda").manual_seed(233),
        image_id=10
    ).images[0]
    evaluate_quality(image, prompt)
    image.save("astronaut.png")
else:
    image = Image.open("astronaut.png")
    patch_map = get_patch_map([image])
    prompt="Astronaut in a desert, warm color palette, muted colors, detailed, 8k"
    image = pipeline(
        prompt=prompt,
        generator=torch.Generator(device="cuda").manual_seed(233),
        image_id=10,
        num_inference_steps=55,
        patch_map=patch_map.tolist(),
    ).images[0]
    evaluate_quality(image, prompt)
    image.save("astronaut-desert.png")
