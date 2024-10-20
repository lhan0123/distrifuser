import torch
import json
import os

from distrifuser.pipelines import DistriSDPipeline
from distrifuser.utils import DistriConfig

distri_config = DistriConfig(height=512, width=512, warmup_steps=4, mode="stale_gn", parallelism="tensor")
pipeline = DistriSDPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
)

# Used for scoring quality
prompt = "astronaut in a desert, cold color palette, muted colors, detailed, 8k"
image_name = "astronaut.png"
captions_file_path = "results/captions.json"

os.makedirs('results', exist_ok=True)
if os.path.exists(captions_file_path):
    with open(captions_file_path, 'r') as json_file:
        data = json.load(json_file)
else:
    data = {}

data[image_name.split('.')[0]] = prompt

with open(captions_file_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
image = pipeline(
    prompt=prompt,
    generator=torch.Generator(device="cuda").manual_seed(233),
    num_inference_steps=35,
).images[0]
if distri_config.rank == 0:
    image.save(os.path.join('results', image_name))
