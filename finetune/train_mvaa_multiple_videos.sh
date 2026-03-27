#!/usr/bin/env bash
#SBATCH --job-name=mvaa
#SBATCH --time=48:00:00
#SBATCH --open-mode=append
#SBATCH --output=output_mvaa_multivideos_32bs_lr1e-4.log
#SBATCH --error=error_mvaa_multivideos_32bs_lr1e-4.log
#SBATCH --gres=gpu:4
#SBATCH --mem=320G
#SBATCH --cpus-per-task=48

# ======== Job Execution Steps ========

# Navigate to the working directory where your code and virtual environment are located
cd /data/hxza352/projects/MVAA_master

# Activate the Python virtual environment (adjust if it's named differently)
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate MVAA

# export CUDA_VERSION=124
# Prevent tokenizer parallelism issues 
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1

# export CUDA_VISIBLE_DEVICES=0,1
GPU_IDS="0,1,2,3"
# export CUDA_VISIBLE_DEVICES=0
# GPU_IDS="0"

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

# Output Configuration
### kitten
OUTPUT_ARGS=(
    --output_dir "./outputs/train/openvid_random1000/"
    --report_to "tensorboard"
)

### multiple videos - 1000 videos
DATA_ARGS=(
    --data_type "i2v_multiple_frame_multi_videos"
    --data_root "data/"
    --caption_column "prompt_random1000.txt"
    --video_column "videos_random1000.txt"
    # --image_column "data/images_videos_random1000.txt"  # comment this line will use first frame of video as image conditioning
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1
    --data_shift true
    --data_shift_degree 0.5
    # 若 data/cache/video_latent 与 prompt_embeddings 已预计算完，加上下面一行可跳过启动时整库遍历预热
    --skip_latent_cache_warmup
)

# Training Configuration
# 有效 batch = batch_size * GPU数 * gradient_accumulation_steps
# 当前 2 卡: batch_size=2 -> 有效 batch=4; batch_size=4 -> 有效 batch=8
# 若增大 batch_size，建议二选一：
#   (1) 保持总样本数不变: 新 train_steps = 70000 * (4/新有效batch)，lr 可保持 2e-5 或略调大
#   (2) 保持 train_steps 不变: 总样本变多，建议 lr 线性缩放，如 新lr = 2e-5 * (新有效batch/4)
# 例如 batch_size=4 时: (1) train_steps=35000, lr=2e-5 或 2.8e-5; (2) train_steps=70000, lr=4e-5
TRAIN_ARGS=(
    --train_steps 2500 # 若增大 batch_size 且要保持“总看到的数据量”不变，请按上面公式减小
    --seed 42 # random seed
    --batch_size 8
    --gradient_accumulation_steps 1
    --learning_rate 1e-4  # 增大 batch 时可按线性缩放调大，如 default: 2e-5（batch_size=4 且不减少 steps 时）
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory False
    --nccl_timeout 18000
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 100 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
    --resume_from_checkpoint "/data/hxza352/projects/MVAA_master/outputs/train/openvid_random1000/checkpoint-2000/"  # if you want to resume from a checkpoint, otherwise, comment this line
    # --resume_training true
)

# Validation Configuration
video_name=dog_running
# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "./data/"
    --validation_steps 100  # should be multiple of checkpointing_steps
    --validation_prompts "prompts_${video_name}.txt"
    --validation_images "images_videos_${video_name}.txt"
    --validation_videos "videos_${video_name}.txt"
    --gen_fps 16
)

# Combine all arguments and launch, training
accelerate launch --main_process_port 29600 --config_file ./finetune/accelerate_config_multigpu.yaml --gpu_ids $GPU_IDS finetune/train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${LORA_ARGS[@]}"

# ########### reduce GPU
# accelerate launch --config_file ./finetune/accelerate_config.yaml --gpu_ids $GPU_IDS finetune/train.py \
#   "${MODEL_ARGS[@]}" \
#   "${OUTPUT_ARGS[@]}" \
#   "${DATA_ARGS[@]}" \
#   "${TRAIN_ARGS[@]}" \
#   "${SYSTEM_ARGS[@]}" \
#   "${CHECKPOINT_ARGS[@]}" \
#   "${VALIDATION_ARGS[@]}" \
#   "${LORA_ARGS[@]}"

