import json
import re
import torch

import sys
from distrifuser.models.distri_clipscore import evaluate_quality
from distrifuser.pipelines import CachedSDXLPipeline
from distrifuser.utils import DistriConfig
from PIL import Image
import numpy as np

def main():
    distri_config = DistriConfig()
    pipeline = CachedSDXLPipeline.from_pretrained(
        distri_config,
        pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        variant="fp16",
        use_safetensors=True,
    )


    pipeline.set_progress_bar_config()

    with open("cluster_labels.json", "r") as f:
        clusters = json.load(f)

    cluster_id = 42
    
    cluster = [int(re.findall(r'\d+', path)[0]) for path in clusters[str(cluster_id)][:100]]

    patch_map = np.loadtxt(f"clusters/cluster_{cluster_id}/patch_map.txt", delimiter=',').astype(int)
    prompt="robot cat designed by minecraft"
    image = pipeline(
        prompt=prompt,
        generator=torch.Generator(device="cuda").manual_seed(233),
        cluster=cluster,
        patch_map=patch_map.tolist(),
    ).images[0]
    evaluate_quality(image, prompt)
    image.save("robot-cat.png")

if __name__ == "__main__":
    main()