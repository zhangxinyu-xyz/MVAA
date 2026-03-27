#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

music_name="Rocket_man"
video_name="dog_running"
prompt="A golden retriever, sporting sleek black sunglasses, with its lengthy fur flowing in the breeze, sprints playfully across a rooftop terrace, recently refreshed by a light rain. The scene unfolds from a distance, the dog's energetic bounds growing larger as it approaches the camera, its tail wagging with unrestrained joy, while droplets of water glisten on the concrete behind it. The overcast sky provides a dramatic backdrop, emphasizing the vibrant golden coat of the canine as it dashes towards the viewer."
music_path="./data/musics/${music_name}.mp3"
video_path="./data/videos/${video_name}.mp4"
music_beats_to_video_motion_path="./data/music_beats_to_video_motion/${video_name}/${music_name}.csv"
input_image_path="./data/videos/${video_name}/frame_0000.png"

lora_path="./models/<your_model_path>/"
output_video_path="./outputs/${video_name}/${music_name}/"


python inference/cli_demo_v2v_interpolation.py \
    --input_image_path $input_image_path \
    --prompt "${prompt}" \
    --model_id "THUDM/CogVideoX-5b-I2V" \
    --lora_scale $lora_scale \
    --output_video_path ${output_video_path} \
    --lora_path 0.3 \
    --music_beats_to_video_motion_path ${music_beats_to_video_motion_path} \
    --music_path ${music_path} \
    --video_path ${video_path}
