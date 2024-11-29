import sys
from qdrant_client import QdrantClient
from utils.patching import BASE_COLLECTION_NAME

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

def main():
    orig_patch_size = sys.argv[1]
    target_patch_size = sys.argv[2]
    
    client = QdrantClient(url="http://localhost:6333")
    
    if not client.collection_exists(f"{BASE_COLLECTION_NAME}_{orig_patch_size}"):
        print(f"Collection {BASE_COLLECTION_NAME}_{orig_patch_size} does not exist.")
        return
    
    image_patches = {}
    client.scroll(f"{BASE_COLLECTION_NAME}_{orig_patch_size}")
    
    if not client.collection_exists(f"{BASE_COLLECTION_NAME}_{target_patch_size}"):
        client.create_collection(f"{BASE_COLLECTION_NAME}_{target_patch_size}")

if __name__ == "__main__":
    main()