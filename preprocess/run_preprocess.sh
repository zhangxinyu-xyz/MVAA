#!/bin/bash

data_dir="./data/"

############ Extract video motion - music beats alignment ############
video_paths="${data_dir}/videos.txt"
music_paths="${data_dir}/musics.txt"

while IFS= read -r video_line
do
    video_path="${data_dir}/${video_line}"
    echo "Input_path: ${video_path}"
    video_filename=$(basename "$video_line")
    echo "Video name: ${video_filename}"

    frames_dir="${data_dir}/${video_line//.mp4}"
    echo "Frames dir: ${frames_dir}"
    motion_dir="${data_dir}/video_motion_peaks/${video_filename//.mp4/}"
    echo "Motion dir: ${motion_dir}"

    # Extract images from videos
    python preprocess/video_tools.py \
        --input_path $video_path \
        --output_dir $frames_dir \
        --type "mp4_to_images" \

    while IFS= read -r music_line
    do
        music_path="${data_dir}/${music_line}"
        echo "Music path: ${music_path}"
        music_filename=$(basename "$music_line")
        echo "Music name: ${music_filename}"
        beats_dir="${data_dir}/music_beats/${music_filename//.mp3/}"
        echo "Music beats dir: ${beats_dir}"

        music_beats_to_video_motion_path="${data_dir}/music_beats_to_video_motion/${video_filename}/${music_filename}"
        echo "Motion-Beats path: ${music_beats_to_video_motion_path}"

        python preprocess/motion_music_alignment.py \
                --video ${video_path} \
                --audio ${music_path} \
                --output ${music_beats_to_video_motion_path}

    done < "$music_paths"
done < "$video_paths"
############################################################
