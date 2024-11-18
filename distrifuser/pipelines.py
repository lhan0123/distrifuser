import inspect
import math
from typing import List, Optional, Union
from qdrant_client import QdrantClient
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

from distrifuser.models.base_unet import BaseUNet
from utils.patching import create_patch_db_if_not_exists, create_prompt_db_if_not_exists, find_similar_prompt, initialize_latents_from_cache, save_patch_to_db, save_prompt_to_db
from utils.patching import DEFAULT_K

from .models.distri_sdxl_unet_pp import DistriUNetPP
from .models.distri_sdxl_unet_tp import DistriUNetTP
from .models.naive_patch_sdxl import NaivePatchUNet
from .utils import DistriConfig, PatchParallelismCommManager
import numpy as np

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    k: int = 0,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps + k, device=device, **kwargs)
        timesteps = scheduler.timesteps[k:]
    return timesteps, num_inference_steps

class CachedSDXLPipeline:
    def __init__(self, pipeline: StableDiffusionXLPipeline, distri_config: DistriConfig):
        self.pipeline = pipeline
        self.distri_config = distri_config
        self.is_profile = distri_config.is_profile

        self.static_inputs = None

        self.client = QdrantClient(url="http://localhost:6333")
        self.prepare()

    @staticmethod
    def from_pretrained(distri_config: DistriConfig, **kwargs):
        device = distri_config.device
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0"
        )
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="unet"
        ).to(device)

        unet = BaseUNet(unet)

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unet, **kwargs
        ).to(device)
        return CachedSDXLPipeline(pipeline, distri_config)

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(self, 
        prompt: Union[str, List[str]],
        guidance_scale: float = 5.0,
        guidance_rescale: float = 0.0,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        image_id: int = -1,
        patch_map: list[list[int]] = None
    ):
        config = self.distri_config
        pipeline = self.pipeline
        
        do_classifier_free_guidance = guidance_scale > 1 and pipeline.unet.config.time_cond_proj_dim is None
        
        if patch_map is None:
            k = DEFAULT_K
        else:
            k = math.floor(np.mean(patch_map))
        
        print(f"{k=}")

        if not self.is_profile:
            num_inference_steps -= k

        height = config.height
        width = config.width

        original_size = (height, width)
        target_size = (height, width)

        # 1. Check inputs. Raise error if not correct
        pipeline.check_inputs(
            prompt,
            None,
            height,
            width,
            None,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = config.device

        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            prompt=prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_inference_steps, device, None, k=0 if self.is_profile else k)

        # 5. Prepare latent variables
        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            None,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, 0.0)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

        add_time_ids = pipeline._get_add_time_ids(
            original_size,
            (0, 0),
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if pipeline.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size)
            timestep_cond = pipeline.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # patching
        create_prompt_db_if_not_exists(self.client, prompt_embeds)
        create_patch_db_if_not_exists(self.client, latents)
        
        if self.is_profile:
            save_prompt_to_db(self.client, prompt_embeds, image_id)
        else:
            image_id = find_similar_prompt(self.client, prompt_embeds)
        
        if not self.is_profile:
            latents = initialize_latents_from_cache(self.client, latents, image_id, patch_map)
            
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast

            if needs_upcasting:
                pipeline.upcast_vae()
                latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)

            image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                pipeline.vae.to(dtype=torch.float16)

            # apply watermark if available
            if pipeline.watermark is not None:
                image = pipeline.watermark.apply_watermark(image)

            image = pipeline.image_processor.postprocess(image)
            image[0].save("cached.png")

        pipeline._num_timesteps = len(timesteps)
        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=pipeline.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                if i + 1 in (5, 10, 15, 20, 25, 30) and self.is_profile:
                    save_patch_to_db(self.client, latents, i + 1, image_id)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                    progress_bar.update()

        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast

        if needs_upcasting:
            pipeline.upcast_vae()
            latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)

        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            pipeline.vae.to(dtype=torch.float16)

        # apply watermark if available
        if pipeline.watermark is not None:
            image = pipeline.watermark.apply_watermark(image)

        image = pipeline.image_processor.postprocess(image)

        # Offload all models
        pipeline.maybe_free_model_hooks()

        return StableDiffusionXLPipelineOutput(images=image)

    @torch.no_grad()
    def prepare(self, **kwargs):
        distri_config = self.distri_config

        static_inputs = {}
        static_outputs = []
        cuda_graphs = []
        pipeline = self.pipeline

        height = distri_config.height
        width = distri_config.width
        assert height % 8 == 0 and width % 8 == 0

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)

        device = distri_config.device

        prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
            prompt="",
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
        )
        batch_size = 2 if distri_config.do_classifier_free_guidance else 1

        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size, num_channels_latents, height, width, prompt_embeds.dtype, device, None
        )

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

        add_time_ids = pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(1, 1)

        if batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
            add_text_embeds = add_text_embeds.repeat(batch_size, 1)
            add_time_ids = add_time_ids.repeat(batch_size, 1)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        t = torch.zeros([batch_size], device=device, dtype=torch.long)

        static_inputs["sample"] = latents
        static_inputs["timestep"] = t
        static_inputs["encoder_hidden_states"] = prompt_embeds
        static_inputs["added_cond_kwargs"] = added_cond_kwargs

        # Used to create communication buffer

        # Pre-run
        pipeline.unet.set_counter(0)
        pipeline.unet(**static_inputs, return_dict=False, record=True)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pipeline.unet.set_counter(0)
            output = pipeline.unet(**static_inputs, return_dict=False, record=True)[0]
            static_outputs.append(output)
        cuda_graphs.append(graph)
        pipeline.unet.setup_cuda_graph(static_outputs, cuda_graphs)

        self.static_inputs = static_inputs

class DistriSDXLPipeline:
    def __init__(self, pipeline: StableDiffusionXLPipeline, module_config: DistriConfig):
        self.pipeline = pipeline
        self.distri_config = module_config

        self.static_inputs = None

        self.prepare()

    @staticmethod
    def from_pretrained(distri_config: DistriConfig, **kwargs):
        device = distri_config.device
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0"
        )
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="unet"
        ).to(device)

        if distri_config.parallelism == "patch":
            unet = DistriUNetPP(unet, distri_config)
        elif distri_config.parallelism == "tensor":
            unet = DistriUNetTP(unet, distri_config)
        elif distri_config.parallelism == "naive_patch":
            unet = NaivePatchUNet(unet, distri_config)
        else:
            raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unet, **kwargs
        ).to(device)
        return DistriSDXLPipeline(pipeline, distri_config)

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert "height" not in kwargs, "height should not be in kwargs"
        assert "width" not in kwargs, "width should not be in kwargs"
        config = self.distri_config
        if not config.do_classifier_free_guidance:
            if "guidance_scale" not in kwargs:
                kwargs["guidance_scale"] = 1
            else:
                assert kwargs["guidance_scale"] == 1
        self.pipeline.unet.set_counter(0)
        return self.pipeline(height=config.height, width=config.width, *args, **kwargs)

    @torch.no_grad()
    def prepare(self, **kwargs):
        distri_config = self.distri_config

        static_inputs = {}
        static_outputs = []
        cuda_graphs = []
        pipeline = self.pipeline

        height = distri_config.height
        width = distri_config.width
        assert height % 8 == 0 and width % 8 == 0

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)

        device = distri_config.device

        prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
            prompt="",
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
        )
        batch_size = 2 if distri_config.do_classifier_free_guidance else 1

        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size, num_channels_latents, height, width, prompt_embeds.dtype, device, None
        )

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

        add_time_ids = pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(1, 1)

        if batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
            add_text_embeds = add_text_embeds.repeat(batch_size, 1)
            add_time_ids = add_time_ids.repeat(batch_size, 1)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        t = torch.zeros([batch_size], device=device, dtype=torch.long)

        static_inputs["sample"] = latents
        static_inputs["timestep"] = t
        static_inputs["encoder_hidden_states"] = prompt_embeds
        static_inputs["added_cond_kwargs"] = added_cond_kwargs

        # Used to create communication buffer
        comm_manager = None
        if distri_config.n_device_per_batch > 1:
            comm_manager = PatchParallelismCommManager(distri_config)
            pipeline.unet.set_comm_manager(comm_manager)

            # Only used for creating the communication buffer
            pipeline.unet.set_counter(0)
            pipeline.unet(**static_inputs, return_dict=False, record=True)
            if comm_manager.numel > 0:
                comm_manager.create_buffer()

        # Pre-run
        pipeline.unet.set_counter(0)
        pipeline.unet(**static_inputs, return_dict=False, record=True)

        if distri_config.use_cuda_graph:
            if comm_manager is not None:
                comm_manager.clear()
            if distri_config.parallelism == "naive_patch":
                counters = [0, 1]
            elif distri_config.parallelism == "patch":
                counters = [0, distri_config.warmup_steps + 1, distri_config.warmup_steps + 2]
            elif distri_config.parallelism == "tensor":
                counters = [0]
            else:
                raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")
            for counter in counters:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    pipeline.unet.set_counter(counter)
                    output = pipeline.unet(**static_inputs, return_dict=False, record=True)[0]
                    static_outputs.append(output)
                cuda_graphs.append(graph)
            pipeline.unet.setup_cuda_graph(static_outputs, cuda_graphs)

        self.static_inputs = static_inputs


class DistriSDPipeline:
    def __init__(self, pipeline: StableDiffusionPipeline, module_config: DistriConfig):
        self.pipeline = pipeline
        self.distri_config = module_config

        self.static_inputs = None

        self.prepare()

    @staticmethod
    def from_pretrained(distri_config: DistriConfig, **kwargs):
        device = distri_config.device
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", "CompVis/stable-diffusion-v1-4")
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="unet"
        ).to(device)

        if distri_config.parallelism == "patch":
            unet = DistriUNetPP(unet, distri_config)
        elif distri_config.parallelism == "tensor":
            unet = DistriUNetTP(unet, distri_config)
        elif distri_config.parallelism == "naive_patch":
            unet = NaivePatchUNet(unet, distri_config)
        else:
            raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")

        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unet, **kwargs
        ).to(device)
        pipeline.safety_checker = lambda images, clip_input: (images, [False])
        return DistriSDPipeline(pipeline, distri_config)

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert "height" not in kwargs, "height should not be in kwargs"
        assert "width" not in kwargs, "width should not be in kwargs"
        config = self.distri_config
        if not config.do_classifier_free_guidance:
            if not "guidance_scale" not in kwargs:
                kwargs["guidance_scale"] = 1
            else:
                assert kwargs["guidance_scale"] == 1
        self.pipeline.unet.set_counter(0)
        return self.pipeline(height=config.height, width=config.width, *args, **kwargs)

    @torch.no_grad()
    def prepare(self, **kwargs):
        distri_config = self.distri_config

        static_inputs = {}
        static_outputs = []
        cuda_graphs = []
        pipeline = self.pipeline

        height = distri_config.height
        width = distri_config.width
        assert height % 8 == 0 and width % 8 == 0

        device = distri_config.device

        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            "",
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=kwargs.get("clip_skip", None),
        )

        batch_size = 2 if distri_config.do_classifier_free_guidance else 1

        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size, num_channels_latents, height, width, prompt_embeds.dtype, device, None
        )

        prompt_embeds = prompt_embeds.to(device)

        if batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)

        t = torch.zeros([batch_size], device=device, dtype=torch.long)

        static_inputs["sample"] = latents
        static_inputs["timestep"] = t
        static_inputs["encoder_hidden_states"] = prompt_embeds

        # Used to create communication buffer
        comm_manager = None
        if distri_config.n_device_per_batch > 1:
            comm_manager = PatchParallelismCommManager(distri_config)
            pipeline.unet.set_comm_manager(comm_manager)

            # Only used for creating the communication buffer
            pipeline.unet.set_counter(0)
            pipeline.unet(**static_inputs, return_dict=False, record=True)
            if comm_manager.numel > 0:
                comm_manager.create_buffer()

        # Pre-run
        pipeline.unet.set_counter(0)
        pipeline.unet(**static_inputs, return_dict=False, record=True)

        if distri_config.use_cuda_graph:
            if comm_manager is not None:
                comm_manager.clear()
            if distri_config.parallelism == "naive_patch":
                counters = [0, 1]
            elif distri_config.parallelism == "patch":
                counters = [0, distri_config.warmup_steps + 1, distri_config.warmup_steps + 2]
            elif distri_config.parallelism == "tensor":
                counters = [0]
            else:
                raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")
            for counter in counters:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    pipeline.unet.set_counter(counter)
                    output = pipeline.unet(**static_inputs, return_dict=False, record=True)[0]
                    static_outputs.append(output)
                cuda_graphs.append(graph)
            pipeline.unet.setup_cuda_graph(static_outputs, cuda_graphs)

        self.static_inputs = static_inputs
