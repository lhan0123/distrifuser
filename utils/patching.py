import time
from typing import Optional
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, MultiVectorConfig, MultiVectorComparator
from qdrant_client.models import Filter, FieldCondition, MatchValue, Datatype
import numpy as np


BASE_COLLECTION_NAME = "patching"
PATCH_SIZE = 16
DEFAULT_K = 15

PROMPT_COLLECTION_NAME = BASE_COLLECTION_NAME + "_prompt"
PATCH_COLLECTION_NAME = BASE_COLLECTION_NAME + f"_{PATCH_SIZE}"


def patchify(tensor: torch.Tensor, patch_size: int = PATCH_SIZE):
    b, ch, _, _ = tensor.shape
    patches = tensor.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size)
    patches = patches.reshape(b, ch, -1, patch_size,
                              patch_size).transpose(0, 2).transpose(1, 2)
    return patches


def create_prompt_db_if_not_exists(client: QdrantClient, prompt_embeds: torch.Tensor):
    if not client.collection_exists(PROMPT_COLLECTION_NAME):
        client.create_collection(
            collection_name=PROMPT_COLLECTION_NAME,
            vectors_config=VectorParams(size=prompt_embeds.shape[-1], 
                                        distance=Distance.EUCLID,
                                        multivector_config=MultiVectorConfig(
                                            comparator=MultiVectorComparator.MAX_SIM
                                        ),
                                        datatype=Datatype.FLOAT16
            ),
        )


def create_patch_db_if_not_exists(client: QdrantClient, latents: torch.Tensor):
    b, c, _, _ = latents.shape
    if not client.collection_exists(PATCH_COLLECTION_NAME):
        client.create_collection(
            collection_name=PATCH_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=b*c*PATCH_SIZE*PATCH_SIZE, 
                distance=Distance.EUCLID,
                datatype=Datatype.FLOAT16
            ),
        )


def save_prompt_to_db(client: QdrantClient, prompt_embeds: torch.Tensor, image_id: int):
    client.upsert(PROMPT_COLLECTION_NAME,
                  wait=False,
                  points=[PointStruct(id=time.time_ns(), vector=prompt_embeds.tolist()[0], payload={
                      "image_id": image_id
                  })])


def find_similar_prompt(client, prompt_embeds):
    point = client.query_points(PROMPT_COLLECTION_NAME, query=prompt_embeds.tolist()[0],
                                with_payload=True,
                                with_vectors=True,
                                limit=1,).points[0]
    image_id = point.payload["image_id"]
    score = point.score / prompt_embeds.shape[1]
    # todo: less than a threshold, don't use cache
    return image_id


def initialize_latents_from_cache(client: QdrantClient, latents: torch.Tensor, image_id: int, patch_map: Optional[list[list[int]]] = None):
    patches = patchify(latents)
    for i, patch in enumerate(patches):
        _, _, ih, iw = latents.shape
        _, _, ph, pw = patch.shape

        patch_x = i % (iw // pw)
        patch_y = i // (ih // ph)
        
        k = patch_map[patch_y][patch_x] if patch_map is not None else DEFAULT_K
        
        cached_patch = client.scroll(
            collection_name=PATCH_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="k",
                        match=MatchValue(value=k),
                    ),
                    FieldCondition(
                        key="index",
                        match=MatchValue(value=i),
                    ),
                    FieldCondition(
                        key="image_id",
                        match=MatchValue(value=image_id),
                    )
                ]
            ),
            with_vectors=True,
            limit=1
        )[0]

        if len(cached_patch) == 0:
            continue

        cached_patch = cached_patch[0].vector
        patch_tensor = torch.Tensor(cached_patch).to(torch.float16).reshape(patch.shape)
        
        latents[:, :, patch_y*ph:(patch_y+1)*ph,
                patch_x*pw:(patch_x+1)*pw] = patch_tensor
    return latents


def save_patch_to_db(client: QdrantClient, latents: torch.Tensor, step: int, image_id: int):
    patches = patchify(latents)
    client.upsert(
        collection_name=PATCH_COLLECTION_NAME,
        wait=False,
        points=[PointStruct(id=time.time_ns() + pid, vector=patch.flatten().tolist(), payload={
            "k": step,
            "index": pid,
            "image_id": image_id
        }) for pid, patch in enumerate(patches)]
    )
