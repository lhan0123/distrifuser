import torch
import torchvision
from torchvision.utils import save_image
from torchvision.io import read_image
import torchvision.transforms as T
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

PATCH_SIZE = 512

def patchify(tensor: torch.Tensor):
    b, c, _, _ = tensor.shape
    patches = tensor.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    patches = patches.reshape(b, c, -1, PATCH_SIZE, PATCH_SIZE).transpose(0, 2).transpose(1, 2)
    return patches
    
def get_saliency_map(image):

    # Load a pre-trained model (e.g., ResNet50)
    model = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.eval()
    w, h = image.size
    
    # Load and preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size, if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    # Display the saliency map
    plt.imsave("saliency.png", saliency, cmap='hot')
    plt.axis('off')
    plt.show()


def main():


    img = Image.open("astronaut.png")
    get_saliency_map(img)
    
    # empty_image = torch.ones((1, 3, 1024, 1024), dtype=torch.float16)

    # orig_image = torch.stack([read_image("astronaut.png")]).float() / 255.0

    # patches = patchify(orig_image)

    # for i, patch in enumerate(patches):
    #     _, _, ih, iw = empty_image.shape
    #     _, _, ph, pw = patch.shape
    #     num_patches = len(patches)
        
    #     patch_x = i % (iw // pw)
    #     patch_y = i // (ih // ph)
    #     empty_image[:, :, patch_y*ph:(patch_y+1)*ph, patch_x*pw:(patch_x+1)*pw] = patch

    # save_image(empty_image, "empty.png")

main()