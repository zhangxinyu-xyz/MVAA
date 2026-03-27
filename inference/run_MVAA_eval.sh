#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

data_dir="./data/"

############ MVAA evaluation: generate music-video aligned videos ############
video_paths="${data_dir}/videos.txt"
prompt_paths="${data_dir}/prompts.txt"
music_paths="${data_dir}/musics.txt"

# open two files simutaneously
exec 3< "$prompt_paths"

while IFS= read -r video_line && IFS= read -r prompt_line <&3
do
    video_path="${data_dir}/${video_line}"
    echo "Input_path: ${video_path}"
    video_filename=$(basename "$video_line")
    echo "Video name: ${video_filename}"

    frames_dir="${data_dir}/${video_line//.mp4}"
    echo "Frames dir: ${frames_dir}"
    motion_dir="${data_dir}/video_motion_peaks/${video_filename//.mp4/}"
    echo "Motion dir: ${motion_dir}"

    echo "Prompt: ${prompt_line}"

    while IFS= read -r music_line
    do
        music_path="${data_dir}/${music_line}"
        music_filename=$(basename "$music_line")
        beats_dir="${data_dir}/music_beats/${music_filename//.mp3/}"
        music_beats_to_video_motion_path="${data_dir}/music_beats_to_video_motion/${video_filename}/${music_filename}"

        music_name=${music_filename//.mp3/}
        video_name=${video_filename//.mp4/}
        echo "video_name: ${video_name}"
        echo "music_name: ${music_name}"

        input_image_path=./data/videos/${video_name}/frame_0000.png
        prompt=${prompt_line}
        
        lora_path="./models/<your_model_path>/"   #### need to change to the path of the model
        output_video_path="./models/<your_output_path>/${video_name}/${music_name}/"   #### need to change to the path of the output

        music_path="./data/musics/${music_name}.mp3"
        video_path="./data/videos/${video_name}.mp4"
        music_beats_to_video_motion_path="./data/music_beats_to_video_motion/${video_name}/${music_name}.csv"

        python inference/cli_demo_v2v_interpolation.py \
            --input_image_path $input_image_path \
            --prompt "${prompt}" \
            --model_id "THUDM/CogVideoX-5b-I2V" \
            --lora_scale 0.3 \
            --output_video_path ${output_video_path} \
            --lora_path ${lora_path} \
            --music_beats_to_video_motion_path ${music_beats_to_video_motion_path} \
            --music_path ${music_path} \
            --video_path ${video_path}

    done < "$music_paths"
done < "$video_paths"
# close file descriptor
exec 3<&-
############################################################


############ MVAA evaluation ############
# Run all metrics (BeatALign, LPIPS, clip_score_frame), or select via --metrics
# Example: --metrics beat_align  |  --metrics LPIPS clip_score_frame  |  --metrics all
python inference/run_mvaa_eval.py \
    --raw_input_txt ${music_paths} ${video_paths} \
    --submission_path "./models/<your_output_path>/" \  #### need to change to the path of the model
    --metrics all \

echo "MVAA evaluation completed"
