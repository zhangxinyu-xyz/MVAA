#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
GPU_IDS="0"

video_name=dog_running

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX-5b-I2V"
    --model_name "cogvideox-i2v"  # ["cogvideox-i2v"]
    --model_type "i2v"
    --training_type "lora_multiple_frames"
)

LORA_ARGS=(
    --rank 128
    --lora_alpha 64
)

OUTPUT_ARGS=(
    --output_dir "./outputs/train/openvid_random10/${video_name}_iter200/"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_type "i2v_multiple_frame"
    --data_root "./data/"
    --caption_column "prompts_${video_name}.txt"
    --video_column "videos_${video_name}.txt"
    --image_column "images_videos_${video_name}.txt"  # comment this line will use first frame of video as image conditioning
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1
    --data_shift true
    --data_shift_degree 0.5
)

# Training Configuration
TRAIN_ARGS=(
    --train_steps 200 # number of training epochs
    --seed 42 # random seed
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 0
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 50 # save checkpoint every x steps
    --checkpointing_limit 1 # maximum number of checkpoints to keep, after which the oldest one is deleted
    --lora_weights_path "./models/openvid_random10/checkpoint-8000/"  # if you want to resume from a checkpoint, otherwise, comment this line
)


# Validation Configuration
VALIDATION_ARGS=(
    --do_validation false  # ["true", "false"]
    --validation_dir "./data/"
    --validation_steps 10  # should be multiple of checkpointing_steps
    --validation_prompts "prompts_${video_name}.txt"
    --validation_images "images_videos_${video_name}.txt"
    --validation_videos "videos_${video_name}.txt"
    --gen_fps 16
)



# Combine all arguments and launch training
accelerate launch --config_file ./finetune/accelerate_config.yaml --gpu_ids $GPU_IDS finetune/train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${LORA_ARGS[@]}"
