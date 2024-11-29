import os
from pathlib import Path
import sys
import pandas as pd
import torch
from distrifuser.pipelines import CachedSDXLPipeline
from distrifuser.utils import DistriConfig
import torch
from urllib.request import urlretrieve

table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
urlretrieve(table_url, 'metadata.parquet')
df = pd.read_parquet('metadata.parquet')

distri_config = DistriConfig(is_profile=True)
pipeline = CachedSDXLPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True,
)

def main():
    force = "-f" in sys.argv
    for i, prompt in enumerate(df['prompt'].head(5000)):
        if os.path.exists(f"datasets/data__{i:05}") and not force:
            continue
        
        image = pipeline(prompt=prompt, generator=torch.Generator(
            device="cuda"), image_id=i).images[0]
        Path(f"datasets/data__{i:05}").mkdir(parents=True, exist_ok=True)
        with open(f"datasets/data__{i:05}/prompt.txt", "w") as f:
            f.write(prompt)
        image.save(f"datasets/data__{i:05}/img.png")
        

if __name__ == "__main__":
    main()