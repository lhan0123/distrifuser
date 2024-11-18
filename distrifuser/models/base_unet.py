import time
import torch
from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, QueryRequest
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .base_model import BaseModel

PATCH_SIZE = 16
REPO_NAME = 'patching'

class BaseUNet(BaseModel):  # for Patch Parallelism
    def __init__(self, model: UNet2DConditionModel):
        assert isinstance(model, UNet2DConditionModel)
        super(BaseUNet, self).__init__(model)

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

        b, c, _, _ = sample.shape
        
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

        if not record:
            static_inputs = self.static_inputs

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

            if record:
                if self.static_inputs is None:
                    self.static_inputs = {
                        "sample": sample,
                        "timestep": timestep,
                        "encoder_hidden_states": encoder_hidden_states,
                        "added_cond_kwargs": added_cond_kwargs,
                    }
                self.synchronize()

        if return_dict:
            output = UNet2DConditionOutput(sample=output)
        else:
            output = (output,)

        self.counter += 1
            
        return output

    @property
    def add_embedding(self):
        return self.model.add_embedding
