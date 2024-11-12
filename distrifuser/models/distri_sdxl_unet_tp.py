import time
import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from torch import distributed as dist, nn

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, QueryRequest
from qdrant_client.models import Filter, FieldCondition, MatchValue

from distrifuser.modules.base_module import BaseModule
from .base_model import BaseModel
from ..modules.tp.attention import DistriAttentionTP
from ..modules.tp.conv2d import DistriConv2dTP
from ..modules.tp.feed_forward import DistriFeedForwardTP
from ..modules.tp.resnet import DistriResnetBlock2DTP
from ..utils import DistriConfig

import distrifuser.models.distri_clipscore as distri_clipscore 

prompt = "astronaut in a desert, cold color palette, muted colors, detailed, 8k"

PATCH_SIZE = 16
REPO_NAME = 'patching'

class DistriUNetTP(BaseModel):  # for Patch Parallelism
    def __init__(self, model: UNet2DConditionModel, distri_config: DistriConfig, profile=False):
        assert isinstance(model, UNet2DConditionModel)
        self.profile = profile
        self.image_id = -1
        if distri_config.world_size > 1 and distri_config.n_device_per_batch > 1:
            for name, module in model.named_modules():
                if isinstance(module, BaseModule):
                    continue
                for subname, submodule in module.named_children():
                    if isinstance(submodule, Attention):
                        wrapped_submodule = DistriAttentionTP(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, FeedForward):
                        wrapped_submodule = DistriFeedForwardTP(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, ResnetBlock2D):
                        wrapped_submodule = DistriResnetBlock2DTP(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, nn.Conv2d) and (
                        subname == "conv_out" or "downsamplers" in name or "upsamplers" in name
                    ):
                        wrapped_submodule = DistriConv2dTP(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)

        self.client = QdrantClient(url="http://localhost:6333")
        super(DistriUNetTP, self).__init__(model, distri_config)

    def patchify(self, tensor: torch.Tensor):
        ch = tensor.shape[1]
        patches = tensor.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.reshape(2, ch, -1, PATCH_SIZE, PATCH_SIZE).transpose(0, 2).transpose(1, 2)
        return patches
    
    def set_image_id(self, image_id):
        self.image_id = image_id

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor or float or int,
        encoder_hidden_states: torch.Tensor,
        class_labels: torch.Tensor or None = None,
        timestep_cond: torch.Tensor or None = None,
        attention_mask: torch.Tensor or None = None,
        cross_attention_kwargs: dict[str, any] or None = None,
        added_cond_kwargs: dict[str, torch.Tensor] or None = None,
        down_block_additional_residuals: tuple[torch.Tensor] or None = None,
        mid_block_additional_residual: torch.Tensor or None = None,
        down_intrablock_additional_residuals: tuple[torch.Tensor] or None = None,
        encoder_attention_mask: torch.Tensor or None = None,
        return_dict: bool = True,
        record: bool = False
    ):
        if not record:
            distri_clipscore.evaluate_quality(sample, prompt)

        distri_config = self.distri_config
        b, c, h, w = sample.shape

        assert (
            class_labels is None
            and timestep_cond is None
            and attention_mask is None
            and cross_attention_kwargs is None
            and down_block_additional_residuals is None
            and mid_block_additional_residual is None
            and down_intrablock_additional_residuals is None
            and encoder_attention_mask is None
        )


        if not self.client.collection_exists(REPO_NAME):
            self.client.create_collection(
                collection_name=REPO_NAME,
                vectors_config=VectorParams(size=b*c*PATCH_SIZE*PATCH_SIZE, distance=Distance.COSINE),
            )

        if not record and not self.profile and self.counter == 0:
            patches = self.patchify(sample)
            for i, patch in enumerate(patches):
                cached_patch = self.client.query_points("diffusers", query=patch.flatten().tolist(),
                        query_filter=Filter(
                            must=[FieldCondition(key="index", match=MatchValue(value=i)), FieldCondition(key="k", match=MatchValue(value=i))]
                        ),
                        with_payload=True,
                        with_vectors=True,
                        limit=1,).points[0].vector
                patch_tensor = torch.Tensor(cached_patch).float().reshape(patch.shape)
                _, _, ih, iw = sample.shape
                _, _, ph, pw = patch.shape
                
                patch_x = i % (iw // pw)
                patch_y = i // (ih // ph)
                sample[:, :, patch_y*ph:(patch_y+1)*ph, patch_x*pw:(patch_x+1)*pw] = patch_tensor

            # results = self.client.query_batch_points("diffusers", requests=[QueryRequest(query=patch.flatten().tolist()
            # , limit=1, with_vector=True) for patch in patches])
            # print(len(results))
            # for result in results:
            #     if result.points[0].score > 0.5:
            #         sample[]
        if distri_config.use_cuda_graph and not record:
            static_inputs = self.static_inputs

            if distri_config.world_size > 1 and distri_config.do_classifier_free_guidance and distri_config.split_batch:
                assert b == 2
                batch_idx = distri_config.batch_idx()
                sample = sample[batch_idx : batch_idx + 1]
                timestep = (
                    timestep[batch_idx : batch_idx + 1] if torch.is_tensor(timestep) and timestep.ndim > 0 else timestep
                )
                encoder_hidden_states = encoder_hidden_states[batch_idx : batch_idx + 1]
                if added_cond_kwargs is not None:
                    for k in added_cond_kwargs:
                        added_cond_kwargs[k] = added_cond_kwargs[k][batch_idx : batch_idx + 1]

            assert static_inputs["sample"].shape == sample.shape
            static_inputs["sample"].copy_(sample)
            if torch.is_tensor(timestep):
                if timestep.ndim == 0:
                    for b in range(static_inputs["timestep"].shape[0]):
                        static_inputs["timestep"][b] = timestep.item()
                else:
                    assert static_inputs["timestep"].shape == timestep.shape
                    static_inputs["timestep"].copy_(timestep)
            else:
                for b in range(static_inputs["timestep"].shape[0]):
                    static_inputs["timestep"][b] = timestep
            assert static_inputs["encoder_hidden_states"].shape == encoder_hidden_states.shape
            static_inputs["encoder_hidden_states"].copy_(encoder_hidden_states)
            if added_cond_kwargs is not None:
                for k in added_cond_kwargs:
                    assert static_inputs["added_cond_kwargs"][k].shape == added_cond_kwargs[k].shape
                    static_inputs["added_cond_kwargs"][k].copy_(added_cond_kwargs[k])

            graph_idx = 0

            self.cuda_graphs[graph_idx].replay()
            output = self.static_outputs[graph_idx]
        else:
            if distri_config.world_size == 1:
                output = self.model(
                    sample,
                    timestep,
                    encoder_hidden_states,
                    class_labels=class_labels,
                    timestep_cond=timestep_cond,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

            elif distri_config.do_classifier_free_guidance and distri_config.split_batch:
                assert b == 2
                batch_idx = distri_config.batch_idx()
                sample = sample[batch_idx : batch_idx + 1]
                timestep = (
                    timestep[batch_idx : batch_idx + 1] if torch.is_tensor(timestep) and timestep.ndim > 0 else timestep
                )
                encoder_hidden_states = encoder_hidden_states[batch_idx : batch_idx + 1]
                if added_cond_kwargs is not None:
                    new_added_cond_kwargs = {}
                    for k in added_cond_kwargs:
                        new_added_cond_kwargs[k] = added_cond_kwargs[k][batch_idx : batch_idx + 1]
                    added_cond_kwargs = new_added_cond_kwargs
                output = self.model(
                    sample,
                    timestep,
                    encoder_hidden_states,
                    class_labels=class_labels,
                    timestep_cond=timestep_cond,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                if self.output_buffer is None:
                    self.output_buffer = torch.empty((b, c, h, w), device=output.device, dtype=output.dtype)
                if self.buffer_list is None:
                    self.buffer_list = [torch.empty_like(output) for _ in range(2)]
                dist.all_gather(
                    self.buffer_list, output.contiguous(), group=distri_config.split_group(), async_op=False
                )
                torch.cat(self.buffer_list, dim=0, out=self.output_buffer)
                output = self.output_buffer
            else:
                output = self.model(
                    sample,
                    timestep,
                    encoder_hidden_states,
                    class_labels=class_labels,
                    timestep_cond=timestep_cond,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                if self.output_buffer is None:
                    self.output_buffer = torch.empty_like(output)
                self.output_buffer.copy_(output)
                output = self.output_buffer
            if record:
                if self.static_inputs is None:
                    self.static_inputs = {
                        "sample": sample,
                        "timestep": timestep,
                        "encoder_hidden_states": encoder_hidden_states,
                        "added_cond_kwargs": added_cond_kwargs,
                    }
                self.synchronize()


        if self.counter in (4, 9, 14, 19) and not record and self.profile:
            patches = self.patchify(output)
            self.client.upsert(
                collection_name=REPO_NAME,
                wait=True,
                points=[PointStruct(id=time.time_ns() + i, vector=patch.flatten().tolist(), payload={
                    "k": self.counter,
                    "index": i,
                    "image_id": self.image_id
                }) for i, patch in enumerate(patches)]
            )
        if return_dict:
            output = UNet2DConditionOutput(sample=output)
        else:
            output = (output,)

        self.counter += 1
            
        return output

    @property
    def add_embedding(self):
        return self.model.add_embedding
