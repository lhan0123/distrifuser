from pathlib import Path
import pandas as pd
import torch
from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig
import torch
from urllib.request import urlretrieve

REPO_NAME = 'diffusers'
PATCH_SIZE = 16

table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
urlretrieve(table_url, 'metadata.parquet')
df = pd.read_parquet('metadata.parquet')

# Read the table using Pandas
# metadata_df = pd.read_parquet('metadata.parquet')
# splits = {'train': 'data/train.parquet', 'test': 'data/eval.parquet'}
# df = pd.read_parquet(
#     "hf://datasets/Gustavosta/Stable-Diffusion-Prompts/" + splits["test"])

distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4, parallelism="tensor")
pipeline = DistriSDXLPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True,
    profile=True
)

def main():
    for i, prompt in enumerate(df['prompt'].head(10000)):
        image = pipeline(prompt=prompt, generator=torch.Generator(
            device="cuda"), image_id=i).images[0]
        Path(f"datasets/data__{i:05}").mkdir(parents=True, exist_ok=True)
        with open(f"datasets/data__{i:05}/prompt.txt", "w") as f:
            f.write(prompt)
        image.save(f"datasets/data__{i:05}/img.png")
        

if __name__ == "__main__":
    main()