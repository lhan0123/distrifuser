import json
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
import numpy as np
from torchvision import models, transforms
import torchvision
import torch
from PIL import Image
import skimage
from torch import nn

from utils.patching import PATCH_SIZE

def preprocess_image(args):
    image, preprocess, device = args
    is_path = type(image) == str
    if is_path:
        image = Image.open(image)
    # Load and preprocess the image
    input_tensor = preprocess(image).to(device)
    if is_path:
        image.close()
    return input_tensor

def get_saliency_maps(images):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load a pre-trained model (e.g., ResNet50)
    model = models.resnet50(
        weights=torchvision.models.ResNet50_Weights.DEFAULT).to(device)
    model.eval()
    
    if type(images) != list:
        images = [images]

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size, if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    pool = ThreadPool(processes=12)
    input_tensors = torch.stack(pool.map(preprocess_image, [(image, preprocess, device) for image in images[:100]])).requires_grad_()
    pool.close()
    
    # Forward pass
    outputs = model(input_tensors)
    scores, _ = torch.max(outputs, dim=1)

    # Backpropagate to get gradients
    model.zero_grad()
    scores.backward(torch.ones_like(scores))

    # Get saliency map
    saliency, _ = torch.max(input_tensors.grad.data.abs(), dim=1)
    saliency = transforms.Resize((1024, 1024))(saliency)
    saliency = saliency.squeeze()
    
    return saliency

avg_pool = nn.AvgPool2d(PATCH_SIZE, stride=PATCH_SIZE)

def get_patch_map(images, patch_size=PATCH_SIZE):
    saliency_maps = get_saliency_maps(images)

    avg_saliency = torch.mean(saliency_maps, dim=0).cpu().numpy()
    patch_avg_saliency = skimage.measure.block_reduce(
        avg_saliency, (patch_size, patch_size), np.mean)

    ks = (20, 15, 10, 5)
    _, bins = np.histogram(patch_avg_saliency, bins=len(ks))
    def mapper(x):
        for i in range(1, len(bins)):
            if x < bins[i]:
                return ks[i-1]
        
        return ks[-1]

    vfunc = np.vectorize(mapper)
    patch_map = vfunc(patch_avg_saliency)

    return patch_map

def main():
    
    with open("cluster_labels.json", "r") as f:
        cluster_labels = json.load(f)
    
    for label, cluster in cluster_labels.items():
        if label == "-1":
            continue
        
        patch_map = get_patch_map(cluster)
        Path(f"clusters/cluster_{label}").mkdir(parents=True, exist_ok=True)
        for image_path in cluster:
            if not Path(f"clusters/cluster_{label}/{image_path.split('/')[1]}.png").exists():
                Path(f"clusters/cluster_{label}/{image_path.split('/')[1]}.png").symlink_to(Path(image_path).absolute())
        np.savetxt(f'clusters/cluster_{label}/patch_map.txt', patch_map, delimiter=',')

if __name__ == "__main__":
    main()