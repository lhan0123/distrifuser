import json
import os
import torch
import clip
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
    
# Function to load and preprocess images
def get_image_embeddings(image_paths):
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    images = [preprocess(Image.open(path)).unsqueeze(0).to(device) for path in image_paths]
    with torch.no_grad():
        image_embeddings = [model.encode_image(image).cpu().numpy() for image in images]
    return np.vstack(image_embeddings)

def main():


    _, dirs, _ = next(os.walk("datasets"))
    # Load and preprocess images
    image_paths = [f"datasets/{dir}/img.png" for dir in dirs]  # Replace with your image paths
    
    embeddings = get_image_embeddings(image_paths)
    
    embeddings = StandardScaler().fit_transform(embeddings)

    similarity_threshold = 0.3  # Adjust this to tune cluster tightness
    min_samples = 5  # Minimum points in a cluster; adjust based on dataset size
    db = DBSCAN(eps=similarity_threshold, min_samples=min_samples, metric="cosine").fit(embeddings)

    # Retrieve cluster labels
    labels = db.labels_

    # Organize images by cluster
    clusters = {}
    for idx, label in enumerate(labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(image_paths[idx])
    
    with open("cluster_labels.json", "w") as f:
        json.dump(clusters, f)


if __name__ == "__main__":
    main()