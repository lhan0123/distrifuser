import time
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, MultiVectorConfig, MultiVectorComparator
from qdrant_client.models import Filter, FieldCondition, MatchValue


REPO_NAME = "patching"
PATCH_SIZE = 16


def patchify(tensor: torch.Tensor, patch_size: int = PATCH_SIZE):
    b, ch, _, _ = tensor.shape
    patches = tensor.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size)
    patches = patches.reshape(b, ch, -1, patch_size,
                              patch_size).transpose(0, 2).transpose(1, 2)
    return patches


def create_prompt_db_if_not_exists(client, collection_name, prompt_embeds):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=prompt_embeds.shape[-1], distance=Distance.COSINE,
                                        multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            ),
            ),
        )


def save_prompt_to_db(client, collection_name, prompt_embeds, image_id):
    create_prompt_db_if_not_exists(client, collection_name, prompt_embeds)
    client.upsert(collection_name,
                  wait=False,
                  points=[PointStruct(id=time.time_ns(), vector=prompt_embeds.tolist()[0], payload={
                      "image_id": image_id
                  })])


def find_similar_prompt(client, collection_name, prompt_embeds):
    create_prompt_db_if_not_exists(client, collection_name, prompt_embeds)
    point = client.query_points(collection_name, query=prompt_embeds.tolist()[0],
                                with_payload=True,
                                with_vectors=True,
                                limit=1,).points[0]
    image_id = point.payload["image_id"]
    score = point.score / prompt_embeds.shape[1]
    # todo: less than a threshold, don't use cache
    return image_id


def initialize_latents_from_cache(client, collection_name, latents):
    patches = patchify(latents)
    for i, patch in enumerate(patches):
        point = client.query_points(collection_name, query=patch.flatten().tolist(),
                                    query_filter=Filter(
            must=[
                FieldCondition(
                    key="k",
                    match=MatchValue(value=15),
                ),
                FieldCondition(
                    key="index",
                    match=MatchValue(value=i),
                ),]
        ),
            with_payload=True,
            with_vectors=True,
            limit=1,).points[0]

        # todo: is score < threshold, don't use cache
        cached_patch = point.vector
        patch_tensor = torch.Tensor(cached_patch).float().reshape(patch.shape)
        _, _, ih, iw = latents.shape
        _, _, ph, pw = patch.shape

        patch_x = i % (iw // pw)
        patch_y = i // (ih // ph)
        latents[:, :, patch_y*ph:(patch_y+1)*ph,
                patch_x*pw:(patch_x+1)*pw] = patch_tensor
    return latents


def save_patch_to_db(client, collection_name, latents, step, image_id):
    patches = patchify(latents)
    client.upsert(
        collection_name=collection_name,
        wait=False,
        points=[PointStruct(id=time.time_ns() + pid, vector=patch.flatten().tolist(), payload={
            "k": step,
            "index": pid,
            "image_id": image_id
        }) for pid, patch in enumerate(patches)]
    )
