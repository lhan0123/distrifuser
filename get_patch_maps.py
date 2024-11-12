import json
from pathlib import Path
import numpy as np
from torchvision import models, transforms
import torchvision
import torch
from PIL import Image
import skimage
import matplotlib.pyplot as plt

PATCH_SIZE = 16

def get_saliency_map(image):

    # Load a pre-trained model (e.g., ResNet50)
    model = models.resnet50(
        weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.eval()
    w, h = image.size

    # Load and preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size, if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    input_tensor.requires_grad_()

    # Forward pass
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()

    # Backpropagate to get gradients
    model.zero_grad()
    output[0, predicted_class].backward()

    # Get saliency map
    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
    saliency = transforms.Resize((w, h))(saliency)
    saliency = saliency.squeeze().cpu().numpy()

    return saliency

def main():
    
    with open("cluster_labels.json", "r") as f:
        cluster_labels = json.load(f)
    
    for label, cluster in cluster_labels.items():
        if label == "-1":
            continue
        saliencies = [get_saliency_map(Image.open(img)) for img in cluster]

        avg_saliency = np.mean(saliencies, axis=0)

        patch_avg_saliency = skimage.measure.block_reduce(
            avg_saliency, (PATCH_SIZE, PATCH_SIZE), np.mean)

        ks = (20, 15, 10, 5)
        _, bins = np.histogram(patch_avg_saliency, bins=len(ks))
        def mapper(x):
            for i in range(1, len(bins)):
                if x < bins[i]:
                    return ks[i-1]
            
            return ks[-1]

        vfunc = np.vectorize(mapper)
        patch_map = vfunc(patch_avg_saliency)
        Path(f"clusters/cluster_{label}").mkdir(parents=True, exist_ok=True)
        np.savetxt(f'clusters/cluster_{label}/patch_map.txt', patch_map, delimiter=',')

if __name__ == "__main__":
    main()