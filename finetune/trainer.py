import sys, os
import random
import hashlib
import json
import logging
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import diffusers
import torch
import transformers
import wandb
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from finetune.constants import LOG_LEVEL, LOG_NAME
from finetune.datasets import I2VDatasetWithResize, T2VDatasetWithResize
from finetune.datasets.utils import (
    load_images,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)
from finetune.schemas import Args, Components, State
from finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)
from pathlib import Path, PosixPath

#### added by xinyu
from finetune.datasets import I2VDatasetMultipleFrameWithResize, I2VDatasetMultipleFrameMultipleVideosWithResize
import numpy as np
from moviepy import *
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import load_file

logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
    "bf16": torch.bfloat16,
}

def convert_config(config):
    """Recursively convert the config dict, ensuring all values are supported types."""
    converted = {}
    for key, value in config.items():
        if isinstance(value, PosixPath):  # Convert PosixPath to str
            converted[key] = str(value)
        elif isinstance(value, tuple):  # Convert tuple to list (if applicable)
            converted[key] = list(value)
        elif isinstance(value, list):  # Recursively convert list
            converted[key] = str([convert_config(item) if isinstance(item, dict) else item for item in value])
        elif value is None:  # Handle None
            converted[key] = "None"  # You can also change it to ""
        elif isinstance(value, dict):  # Recursively convert dict
            converted[key] = convert_config(value)
        else:
            converted[key] = value  # Keep original value
    return converted

class Trainer:
    # If set, should be a list of components to unload (refer to `Components``)
    UNLOAD_LIST: List[str] = None

    def __init__(self, args: Args) -> None:
        self.args = args
        self.state = State(
            weight_dtype=self.__get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
        )

        self.components: Components = self.load_components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()

        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs")
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.accelerator = accelerator

        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self) -> None:
        if self.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)

    def check_setting(self) -> None:
        # Check for unload_list
        if self.UNLOAD_LIST is None:
            logger.warning(
                "\033[91mNo unload_list specified for this Trainer. All components will be loaded to GPU during training.\033[0m"
            )
        else:
            for name in self.UNLOAD_LIST:
                if name not in self.components.model_fields:
                    raise ValueError(f"Invalid component name in unload_list: {name}")

    def prepare_models(self) -> None:
        logger.info("Initializing models")

        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.transformer.config

    def _get_missing_cache_indices(self) -> List[int]:
        """Return dataset indices whose video latent or prompt embedding cache is missing."""
        train_resolution_str = "x".join(str(x) for x in self.args.train_resolution)
        cache_dir = Path(self.args.data_root) / "cache"
        video_latent_dir = cache_dir / "video_latent" / self.args.model_name / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"

        videos = getattr(self.dataset, "videos", [])
        prompts = getattr(self.dataset, "prompts", [])

        missing = []
        for i in range(len(videos)):
            video_cached = (
                video_latent_dir.exists()
                and (video_latent_dir / (Path(videos[i]).stem + ".safetensors")).exists()
            )
            prompt_cached = (
                prompt_embeddings_dir.exists()
                and (
                    prompt_embeddings_dir
                    / (hashlib.sha256(prompts[i].encode()).hexdigest() + ".safetensors")
                ).exists()
            )
            if not video_cached or not prompt_cached:
                missing.append(i)

        return missing

    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        if self.args.model_type == "i2v":
            if self.args.data_type is not None and self.args.data_type == "i2v_multiple_frame":
                self.dataset = I2VDatasetMultipleFrameWithResize(
                    **(self.args.model_dump()),
                    device=self.accelerator.device,
                    max_num_frames=self.state.train_frames,
                    height=self.state.train_height,
                    width=self.state.train_width,
                    trainer=self,
                )
            elif self.args.data_type is not None and self.args.data_type == "i2v_multiple_frame_multi_videos":
                self.dataset = I2VDatasetMultipleFrameMultipleVideosWithResize(
                    **(self.args.model_dump()),
                    device=self.accelerator.device,
                    max_num_frames=self.state.train_frames,
                    height=self.state.train_height,
                    width=self.state.train_width,
                    trainer=self,
                )
            else:
                self.dataset = I2VDatasetWithResize(
                    **(self.args.model_dump()),
                    device=self.accelerator.device,
                    max_num_frames=self.state.train_frames,
                    height=self.state.train_height,
                    width=self.state.train_width,
                    trainer=self,
                )
        elif self.args.model_type == "t2v":
            self.dataset = T2VDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                trainer=self,
            )
        else:
            raise ValueError(f"Invalid model type: {self.args.model_type}")

        # Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )

        # One full pass to fill cache/video_latent and cache/prompt_embeddings on disk (see dataset __getitem__).
        # If caches already exist, each step only reads safetensors — but this loop still iterates the whole
        # dataset unless --skip_latent_cache_warmup (then we unload VAE/TE immediately).
        if self.args.skip_latent_cache_warmup:
            logger.info(
                "Skipping latent cache warmup (--skip_latent_cache_warmup). "
                "Ensure cache/video_latent/ and cache/prompt_embeddings/ under data_root are complete, "
                "or the first training batches will encode missing entries."
            )
        else:
            missing_indices = self._get_missing_cache_indices()
            total = len(getattr(self.dataset, "videos", []))
            if len(missing_indices) == 0:
                logger.info(
                    f"All {total} video latent and prompt embedding caches already exist. "
                    "Skipping precompute."
                )
            else:
                logger.info(
                    f"Precomputing latent cache for {len(missing_indices)}/{total} missing samples ..."
                )
                for idx in missing_indices:
                    _ = self.dataset[idx]
                self.accelerator.wait_for_everyone()
                logger.info("Precomputing latent cache ... Done")

        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and attr_name == "transformer":
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)

        if self.args.training_type == "lora" or self.args.training_type == "lora_multiple_frames":
            # Prefer lora_weights_path if provided; otherwise fall back to resume_from_checkpoint (backward compatible)
            ckpt_dir = self.args.lora_weights_path or self.args.resume_from_checkpoint
            lora_weights_path = None if ckpt_dir is None else os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors")

            if lora_weights_path is not None and os.path.exists(lora_weights_path):
                print(f"🔄 Loading LoRA weights from: {lora_weights_path}")

                transformer_lora_config = LoraConfig(
                    r=self.args.rank,
                    lora_alpha=self.args.lora_alpha,
                    init_lora_weights=False,
                    target_modules=self.args.target_modules,
                )
                self.components.transformer.add_adapter(transformer_lora_config)

                state_dict = load_file(lora_weights_path)
                self.components.transformer.load_state_dict(state_dict, strict=False)
                self.__prepare_saving_loading_hooks(transformer_lora_config)
                print(f"✅ LoRA weights loaded")
            else:
                transformer_lora_config = LoraConfig(
                    r=self.args.rank,
                    lora_alpha=self.args.lora_alpha,
                    init_lora_weights=True,
                    target_modules=self.args.target_modules,
                )
                self.components.transformer.add_adapter(transformer_lora_config)
                self.__prepare_saving_loading_hooks(transformer_lora_config)

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = ["transformer"] + self.UNLOAD_LIST
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # Make sure the trainable params are in float32
        cast_training_params([self.components.transformer], dtype=torch.float32)

        # For LoRA, we only want to train the LoRA weights
        # For SFT, we want to train all the parameters
        trainable_parameters = list(filter(lambda p: p.requires_grad, self.components.transformer.parameters()))
        transformer_parameters_with_lr = {
            "params": trainable_parameters,
            "lr": self.args.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_for_validation(self):
        validation_prompts = load_prompts(self.args.validation_dir / self.args.validation_prompts)

        if self.args.validation_images is not None:
            validation_images = load_images(self.args.validation_dir / self.args.validation_images, data_dir=self.args.validation_dir)
            if self.args.data_type == 'i2v_multiple_frame' or self.args.data_type == 'i2v_multiple_frame_multi_videos':
                validation_images = [validation_images] * len(validation_prompts)
        else:
            validation_images = [None] * len(validation_prompts)

        if self.args.validation_videos is not None:
            validation_videos = load_videos(self.args.validation_dir / self.args.validation_videos, data_dir=self.args.validation_dir)
            validation_videos = [self.args.validation_dir / p for p in validation_videos]
        else:
            validation_videos = [None] * len(validation_prompts)

        self.state.validation_prompts = validation_prompts
        self.state.validation_images = validation_images
        self.state.validation_videos = validation_videos

    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name or "finetrainers-experiment"
        self.accelerator.init_trackers(tracker_name)
        # self.accelerator.init_trackers(tracker_name, config=convert_config(self.args.model_dump()))

    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )

        ### modified by Xinyu
        if self.args.resume_training:
            try:
                if resume_from_checkpoint_path is not None:
                    self.accelerator.load_state(resume_from_checkpoint_path)
                    logger.info(f"✅ Resumed training state from: {resume_from_checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to resume from {resume_from_checkpoint_path}: {e}")
        else:
            logger.info("Not resuming training state (resume_training=False).")

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss = self.compute_loss(batch)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.transformer.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)

                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)

                # Maybe run validation
                should_run_validation = self.args.do_validation and global_step % self.args.validation_steps == 0
                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    def validate(self, step: int) -> None:
        logger.info("Starting validation")
        
        accelerator = self.accelerator
        num_validation_samples = len(self.state.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()

        if self.state.using_deepspeed:
            # Can't using model_cpu_offload in deepspeed,
            # so we need to move all components in pipe to device
            # pipe.to(self.accelerator.device, dtype=self.state.weight_dtype)
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=["transformer"])
        else:
            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
            # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
            pipe.enable_model_cpu_offload(device=self.accelerator.device)

            # Convert all model weights to training dtype
            # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
            pipe = pipe.to(dtype=self.state.weight_dtype)

        #################################

        all_processes_artifacts = []
        for i in range(num_validation_samples):
            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
                # Skip current validation on all processes but one
                if i % accelerator.num_processes != accelerator.process_index:
                    continue

            prompt = self.state.validation_prompts[i]
            image = self.state.validation_images[i]
            video = self.state.validation_videos[i]

            if image is not None:
                if self.args.data_type == "i2v_multiple_frame"  or self.args.data_type == "i2v_multiple_frame_multi_videos":
                    select_image_index = list(range(len(image))[::8]) 
                    candidate_inserted_position = [
                        select_image_index,
                        [_idx + 4 if _idx != 0 else 0 for _idx in select_image_index] 
                    ]
                    select_image_insert_index = random.choice(candidate_inserted_position)
                    select_image_insert_index = [_idx // self.components.vae.config.temporal_compression_ratio for _idx in select_image_insert_index]
                    image = [preprocess_image_with_resize(_image, self.state.train_height, self.state.train_width) for _image in image]
                    image = [Image.fromarray(_image.to(torch.uint8).permute(1, 2, 0).cpu().numpy()) for _image in image]
                else:
                    image = preprocess_image_with_resize(image, self.state.train_height, self.state.train_width)
                    # Convert image tensor (C, H, W) to PIL images
                    image = image.to(torch.uint8)
                    image = image.permute(1, 2, 0).cpu().numpy()
                    image = Image.fromarray(image)

            if video is not None:
                video = preprocess_video_with_resize(
                    video, self.state.train_frames, self.state.train_height, self.state.train_width
                )
                # Convert video tensor (F, C, H, W) to list of PIL images
                video = video.round().clamp(0, 255).to(torch.uint8)
                video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]

            logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )

            if self.args.data_type == "i2v_multiple_frame"  or self.args.data_type == "i2v_multiple_frame_multi_videos":
                validation_artifacts = self.validation_step({"prompt": prompt, "image": image, "video": video}, pipe, 
                                                            select_image_index=select_image_index, select_image_insert_index=select_image_insert_index,
                                                            num_inference_steps=self.args.validation_num_inference_steps)
            else:
                validation_artifacts = self.validation_step({"prompt": prompt, "image": image, "video": video}, pipe,
                                                            num_inference_steps=self.args.validation_num_inference_steps)

            if (
                self.state.using_deepspeed
                and self.accelerator.deepspeed_plugin.zero_stage == 3
                and not accelerator.is_main_process
            ):
                continue

            prompt_filename = string_to_filename(prompt)[:25]
            # Calculate hash of reversed prompt as a unique identifier
            reversed_prompt = prompt[::-1]
            hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5]

            artifacts = {
                "image": {"type": "image", "value": image},
                "video": {"type": "video", "value": video},
            }
            for i, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                if artifact_type == "select_image_index" or artifact_type == "select_image_insert_index":
                    artifacts.update({f"{artifact_type}": {"type": artifact_type, "value": artifact_value}})
                else:
                    artifacts.update({f"artifact_{i}": {"type": artifact_type, "value": artifact_value}})
            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            # accelerator.log(logs, step=global_step)
            for key, value in list(artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type not in ["image", "video"] or artifact_value is None:
                    continue

                extension = "png" if artifact_type == "image" else "mp4"
                if self.args.data_type == 'i2v_multiple_frame' or self.args.data_type == "i2v_multiple_frame_multi_videos":
                    if artifact_type == "video":
                        if key == "video":
                            filename = f"validation-{accelerator.process_index}-{prompt_filename}-srcvideo.{extension}"
                        else:
                            filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}-{'_'.join(map(str, artifacts['select_image_index']['value']))}.{extension}"
                    else:
                        filename = f"validation-{accelerator.process_index}-{prompt_filename}-{'_'.join(map(str, artifacts['select_image_index']['value']))}.{extension}"
                    validation_path = self.args.output_dir / "validation_res"
                    validation_path.mkdir(parents=True, exist_ok=True)
                    filename = str(validation_path / filename)
                    
                    if artifact_type == "image":
                        logger.debug(f"Saving image to {filename}")
                        width, height = artifact_value[0].size
                        new_image = Image.new('RGB', (width * 2, height * 2))
                        new_image.paste(artifact_value[artifacts["select_image_index"]["value"][0]], (0, 0))
                        new_image.paste(artifact_value[artifacts["select_image_index"]["value"][1]], (width, 0))
                        new_image.paste(artifact_value[artifacts["select_image_index"]["value"][2]], (0, height))
                        new_image.paste(artifact_value[artifacts["select_image_index"]["value"][3]], (width, height))
                        if not os.path.exists(filename):
                            new_image.save(filename)
                        if accelerator.trackers and accelerator.trackers[0].name == "wandb":
                            artifact_value = wandb.Image(filename)
                        else:
                            artifact_value = np.array(new_image).astype(np.float32) / 255.0
                            artifact_value = (artifact_type, artifact_value)
                    elif artifact_type == "video":
                        logger.debug(f"Saving video to {filename}")
                        if not os.path.exists(filename):
                            export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                        if accelerator.trackers and accelerator.trackers[0].name == "wandb":
                            artifact_value = wandb.Video(filename, caption=prompt)
                        else:
                            artifact_value = [np.array(frame).transpose((2, 0, 1)).astype(np.float32) / 255.0 for frame in artifact_value]
                            artifact_value = (artifact_type, np.stack(artifact_value, axis=0))
                else:
                    filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}-{hash_suffix}.{extension}"
                    validation_path = self.args.output_dir / "validation_res"
                    validation_path.mkdir(parents=True, exist_ok=True)
                    filename = str(validation_path / filename)
                    if artifact_type == "image":
                        logger.debug(f"Saving image to {filename}")
                        artifact_value.save(filename)
                        if accelerator.trackers and accelerator.trackers[0].name == "wandb":
                            artifact_value = wandb.Image(filename)
                        else:
                            artifact_value = np.array(new_image).astype(np.float32) / 255.0
                            artifact_value = (artifact_type, artifact_value)
                    elif artifact_type == "video":
                        logger.debug(f"Saving video to {filename}")
                        export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                        if accelerator.trackers and accelerator.trackers[0].name == "wandb":
                            artifact_value = wandb.Video(filename, caption=prompt)
                        else:
                            artifact_value = [np.array(frame).astype(np.float32) / 255.0 for frame in artifact_value]
                            artifact_value = (artifact_type, np.stack(artifact_value, axis=0))
                        
                all_processes_artifacts.append(artifact_value)

        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            tracker_key = "validation"
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
                    video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
                    tracker.log(
                        {
                            tracker_key: {"images": image_artifacts, "videos": video_artifacts},
                        },
                        step=step,
                    )
                ## added by xinyu: visualize the generated video
                elif tracker.name == "tensorboard":
                    image_artifacts = [artifact[1] for artifact in all_artifacts if artifact[0] == "image"]
                    video_artifacts = [artifact[1] for artifact in all_artifacts if artifact[0] == "video"]
                    for idx, img_artifact in enumerate(image_artifacts):
                        tracker.writer.add_image(f"{tracker_key}/image_{idx}", img_artifact, step, dataformats="HWC")
                    for idx, video_artifact in enumerate(video_artifacts):
                        tracker.writer.add_video(f"{tracker_key}/video_{idx}", video_artifact[1:][None, ...], step, fps=16)
                        tracker.writer.flush()

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            # Unload models except those needed for training
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
            # self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
            # self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)
            # cast_training_params([self.components.transformer], dtype=torch.float32)
        else:
            pipe.remove_all_hooks()
            del pipe
            # Load models except those not needed for training
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)

            # Change trainable weights back to fp32 to keep with dtype after prepare the model
            cast_training_params([self.components.transformer], dtype=torch.float32)

        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.transformer.train()

    def fit(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        if self.args.do_validation:
            self.prepare_for_validation()
        self.prepare_trackers()
        self.train()

    def collate_fn(self, examples: List[Dict[str, Any]]):
        raise NotImplementedError

    def load_components(self) -> Components:
        raise NotImplementedError

    def initialize_pipeline(self) -> DiffusionPipeline:
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W], where B = 1
        # shape of output video: [B, C', F', H', W'], where B = 1
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:
        # shape of output text: [batch size, sequence length, embedding dimension]
        raise NotImplementedError

    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        raise NotImplementedError

    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(self.components, name, component.to(self.accelerator.device, dtype=dtype))

    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list: # or name == "vae":
                    # if component.device.type == "meta":
                    #     setattr(self.components, name, component.to_empty(device=torch.device("cpu")))
                    # else:
                    setattr(self.components, name, component.to("cpu"))

    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        model = unwrap_model(self.accelerator, model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                self.components.pipeline_cls.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        transformer_ = unwrap_model(self.accelerator, model)
                    else:
                        raise ValueError(f"Unexpected save model: {unwrap_model(self.accelerator, model).__class__}")
            else:
                transformer_ = unwrap_model(self.accelerator, self.components.transformer).__class__.from_pretrained(
                    self.args.model_path, subfolder="transformer"
                )
                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = self.components.pipeline_cls.lora_state_dict(input_dir)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def __maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
            if must_save or global_step % self.args.checkpointing_steps == 0:
                # for training
                save_path = get_intermediate_ckpt_path(
                    checkpointing_limit=self.args.checkpointing_limit,
                    step=global_step,
                    output_dir=self.args.output_dir,
                )
                self.accelerator.save_state(save_path, safe_serialization=True)

                if not self.args.save_deepspeed_model:
                    import shutil
                    ds_model_dir = os.path.join(save_path, "pytorch_model")
                    if os.path.isdir(ds_model_dir):
                        shutil.rmtree(ds_model_dir)
                        logger.info(f"Removed DeepSpeed model dir to save disk: {ds_model_dir}")
