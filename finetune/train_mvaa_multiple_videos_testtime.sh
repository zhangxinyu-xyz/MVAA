#!/usr/bin/env bash
#SBATCH --job-name=mvaa-tt
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=output_mvaa_multivideos_32bs_lr1e-4_testtime3.log
#SBATCH --error=error_mvaa_multivideos_32bs_lr1e-4_testtime3.log
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

# ======== Job Execution Steps ========
cd /data/hxza352/projects/MVAA_master

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate MVAA

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1

# SLURM 会自动设置 CUDA_VISIBLE_DEVICES，不需要手动指定
GPU_IDS="0"

# ======== Configuration ========
VIDEO_LIST="data/videos.txt"
PRETRAINED_CKPT="./outputs/train/openvid_random1000/checkpoint-2500/"
BASE_OUTPUT_DIR="./outputs/train/openvid_random1000/testtime"
DATA_ROOT="./data/"

TRAIN_STEPS=60           # test-time training steps per video
CKPT_STEPS=20             # save checkpoint every N steps
VAL_STEPS=20              # validation every N steps (must be multiple of CKPT_STEPS)
BATCH_SIZE=8
LR=4e-5

# ======== Video subset control ========
# START_IDX: 从第几个视频开始 (0-based)
# NUM_VIDEOS: 跑几个视频，设为 -1 表示从 START_IDX 跑到末尾
START_IDX=${START_IDX:-0}
NUM_VIDEOS=${NUM_VIDEOS:--1}

# ======== Read video names from list ========
ALL_NAMES=()
while IFS= read -r line; do
    filename=$(basename "$line" .mp4)
    ALL_NAMES+=("$filename")
done < "$VIDEO_LIST"

TOTAL=${#ALL_NAMES[@]}
if [ "$NUM_VIDEOS" -eq -1 ]; then
    END_IDX=$TOTAL
else
    END_IDX=$((START_IDX + NUM_VIDEOS))
    [ $END_IDX -gt $TOTAL ] && END_IDX=$TOTAL
fi
NAMES=("${ALL_NAMES[@]:$START_IDX:$((END_IDX - START_IDX))}")

echo "========================================"
echo "Total videos: $TOTAL, running index $START_IDX..$((END_IDX-1)) (${#NAMES[@]} videos)"
echo "Videos: ${NAMES[*]}"
echo "========================================"

for NAME in "${NAMES[@]}"; do
    OUTPUT_PATH="${BASE_OUTPUT_DIR}/${NAME}/"

    if [ -d "$OUTPUT_PATH" ]; then
        echo "[SKIP] $OUTPUT_PATH already exists, skipping $NAME"
        continue
    fi

    echo "========================================"
    echo "[START] Test-time training: $NAME"
    echo "========================================"

    MODEL_ARGS=(
        --model_path "THUDM/CogVideoX-5b-I2V"
        --model_name "cogvideox-i2v"
        --model_type "i2v"
        --training_type "lora_multiple_frames"
    )

    LORA_ARGS=(
        --rank 128
        --lora_alpha 64
    )

    OUTPUT_ARGS=(
        --output_dir "${OUTPUT_PATH}"
        --report_to "tensorboard"
    )

    DATA_ARGS=(
        --data_type "i2v_multiple_frame"
        --data_root "${DATA_ROOT}"
        --caption_column "prompts_${NAME}.txt"
        --video_column "videos_${NAME}.txt"
        --image_column "images_videos_${NAME}.txt"
        --train_resolution "49x480x720"
        --data_shift true
        --data_shift_degree 0.5
    )

    TRAIN_ARGS=(
        --train_steps ${TRAIN_STEPS}
        --seed 42
        --batch_size ${BATCH_SIZE}
        --gradient_accumulation_steps 1
        --learning_rate ${LR}
        --mixed_precision "bf16"
    )

    SYSTEM_ARGS=(
        --num_workers 0
        --pin_memory True
        --nccl_timeout 1800
    )

    CHECKPOINT_ARGS=(
        --checkpointing_steps ${CKPT_STEPS}
        --checkpointing_limit 1
        --lora_weights_path "${PRETRAINED_CKPT}"
        --save_deepspeed_model false  # save deepspeed model to save disk, no pytorch_model dir
    )

    VALIDATION_ARGS=(
        --do_validation true
        --validation_dir "${DATA_ROOT}"
        --validation_steps ${VAL_STEPS}
        --validation_prompts "prompts_${NAME}.txt"
        --validation_images "images_videos_${NAME}.txt"
        --validation_videos "videos_${NAME}.txt"
        --validation_num_inference_steps 50
        --gen_fps 16
    )

    accelerate launch --main_process_port 29950 \
        --config_file ./finetune/accelerate_config.yaml \
        --gpu_ids $GPU_IDS \
        finetune/train.py \
        "${MODEL_ARGS[@]}" \
        "${OUTPUT_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${TRAIN_ARGS[@]}" \
        "${SYSTEM_ARGS[@]}" \
        "${CHECKPOINT_ARGS[@]}" \
        "${VALIDATION_ARGS[@]}" \
        "${LORA_ARGS[@]}"

    echo "[DONE] $NAME → $OUTPUT_PATH"
    echo ""
done

echo "========================================"
echo "All test-time training completed."
echo "========================================"
