import pandas as pd

splits = {'train': 'data/train.parquet', 'test': 'data/eval.parquet'}
df = pd.read_parquet("hf://datasets/Gustavosta/Stable-Diffusion-Prompts/" + splits["train"])

print(df)