import torch

from distrifuser.pipelines import DistriSDPipeline
from distrifuser.utils import DistriConfig

distri_config = DistriConfig(height=512, width=512, warmup_steps=4, mode="stale_gn", parallelism="tensor")
pipeline = DistriSDPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
)

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
image = pipeline(
    prompt="Camel in a desert, warm color palette, muted colors, detailed, 8k",
    generator=torch.Generator(device="cuda").manual_seed(233),
    num_inference_steps=35,
).images[0]
if distri_config.rank == 0:
    image.save("camel-desert-cached-50.png")
