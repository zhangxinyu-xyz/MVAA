import argparse
import os
from moviepy.editor import VideoFileClip, ColorClip, TextClip, CompositeVideoClip, clips_array, ImageClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def resize_videos_to_same_height(videos, target_height):
    return [v.resize(height=target_height) for v in videos]


def add_label(clip, label_text):
    box_width = int(clip.w * 0.2)
    box_height = int(clip.h * 0.12)
    padding = 15

    # 背景框
    img = Image.new("RGBA", (box_width, box_height), (173, 216, 230, int(255 * 0.6)))
    draw = ImageDraw.Draw(img)

    # 尝试找到合适的字体大小
    target_text_height = int(box_height * 0.6)  # 目标字体高度（大约占70%高度）
    fontsize = 15
    try:
        while True:
            font = ImageFont.load_default(fontsize) #ImageFont.truetype("DejaVuSans.ttf", fontsize)
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_height = bbox[3] - bbox[1]
            if text_height >= target_text_height - 20 or fontsize > 180:
                break
            fontsize += 1
    except IOError:
        font = ImageFont.load_default()
        print("⚠️ Arial 字体加载失败，使用默认字体")

    # 重新算文字位置（左对齐，垂直居中）
    bbox = draw.textbbox((0, 0), label_text, font=font)
    text_height = bbox[3] - bbox[1]
    text_position = (padding, (box_height - text_height) // 2)

    draw.text(text_position, label_text, font=font, fill="black")

    label_np = np.array(img)
    label_clip = ImageClip(label_np, duration=clip.duration).set_position(("left", "top"))

    return CompositeVideoClip([clip, label_clip])


def create_black_placeholder(width, height, duration):
    return ColorClip(size=(width, height), color=(0, 0, 0)).set_duration(duration)

def side_by_side_multi(video_paths, output_path, rows, cols, labels=None, audio_path=None):
    clips = [VideoFileClip(path) for path in video_paths]
    first_audio = clips[0].audio
    
    # 去掉音频
    clips = [clip.without_audio() for clip in clips]

    # 统一尺寸和时长
    target_height = min([clip.h for clip in clips])
    clips = resize_videos_to_same_height(clips, target_height)
    min_duration = min([clip.duration for clip in clips])
    clips = [clip.subclip(0, min_duration) for clip in clips]

    # 加标签
    if labels:
        while len(labels) < len(clips):
            labels.append("")  # 没有标签就空着
        clips = [add_label(c, t) for c, t in zip(clips, labels)]

    # 自动填满不足格子
    total_slots = rows * cols
    if len(clips) < total_slots:
        width = clips[0].w
        height = clips[0].h
        placeholder = create_black_placeholder(width, height, min_duration)
        clips.extend([placeholder] * (total_slots - len(clips)))

    # 构建网格
    clips_array_list = []
    for i in range(rows):
        row_clips = clips[i * cols:(i + 1) * cols]
        clips_array_list.append(row_clips)

    final_clip = clips_array(clips_array_list)
    
    # if audio_path:
    #     from moviepy.editor import AudioFileClip
    #     audio_clip = AudioFileClip(audio_path).subclip(0, final_clip.duration)
    #     final_clip = final_clip.set_audio(audio_clip)
        
    if audio_path:  # audio_paths 是一个音频路径的列表，比如 ['a.mp3', 'b.mp3']
        from moviepy.editor import AudioFileClip, concatenate_audioclips
        audio_clips = [AudioFileClip(p) for p in audio_path]

        # 拼接所有音频片段
        full_audio = concatenate_audioclips(audio_clips)

        # 如果音频总长比视频短，循环补全；如果长于视频，裁剪
        if full_audio.duration < final_clip.duration:
            loops = int(final_clip.duration // full_audio.duration) + 1
            full_audio = concatenate_audioclips([full_audio] * loops)

        final_audio = full_audio.subclip(0, final_clip.duration)
        final_clip = final_clip.set_audio(final_audio)
    elif first_audio:
        # 设置第一个视频的音频为最终剪辑的音频
        final_clip = final_clip.set_audio(first_audio)
    
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"✅ 拼接完成，输出: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多视频拼接，支持自动补空和标题")
    parser.add_argument("--videos", nargs='+', type=str, help="输入视频路径（多个）")
    parser.add_argument("--audio", nargs='+', type=str, help="音频路径（可选）")
    parser.add_argument("--output", type=str, help="输出视频路径")
    parser.add_argument("--rows", type=int, required=True, help="行数")
    parser.add_argument("--cols", type=int, required=True, help="列数")
    parser.add_argument("--labels", nargs='*', help="每个视频的标签（可选）")

    args = parser.parse_args()

    side_by_side_multi(
        video_paths=args.videos,
        output_path=args.output,
        rows=args.rows,
        cols=args.cols,
        labels=args.labels,
        audio_path=args.audio
    )
    
    #### python preprocess/multi_video_grid.py  --rows 1 --cols 2 --labels "ori video" "edit video" --audio "background_music.mp3" --output /data/xinyu/projects/Audio2Video/resources/christmas_cat_16fps/Jingle_Bells_5s_36-41_christmas-cat-16fps_motion_peaks_to_beats_final_cut_remap_vs_baseline.mp4 --videos "/data/xinyu/projects/Audio2Video/resources/Jingle_Bells_5s_36-41_christmas-cat-16fps.mp4" "/data/xinyu/projects/Audio2Video/resources/christmas_cat_16fps/Jingle_Bells_5s_36-41_christmas-cat-16fps_motion_peaks_to_beats_final_cut_remap.mp4"
    
    
    
    ## python Audio2Video/preprocess/multi_video_grid.py  --rows 1 --cols 3 --labels "ori video" "edit video train1 ft2" "edit video train10" --audio /home/xinyu/projects/CogVideo/Audio2Video/resources/Rocket_man/Rocket_man_3s_56-59.mp3 --output /home/xinyu/projects/CogVideo/outputs/train/cogvideox-5b-i2v/multiple_frame_withval_multiple_videos/samples_dog_running/Rocket_man/A_golden_retriever,_sporting_sleek_black_sunglasses,_with_its_lengthy_fur_flowing_in_the_breeze,_spr_fps16_insert0-19-24-32_position0-5-7-9_lora0.3_withaudio_compare_ft1.mp4 --videos /home/xinyu/projects/CogVideo/resources/videos/dog_running_3s_16fps_480x720.mp4 /home/xinyu/projects/CogVideo/outputs/train/cogvideox-5b-i2v/Jingle_Bells_3s_36-41_christmas-cat-16fps_480x720/multiple_frame_withval/iter9600_withshift0.5/test_time_train_dog_running_iter96/samples_dog_running/Rocket_man/A_golden_retriever,_sporting_sleek_black_sunglasses,_with_its_lengthy_fur_flowing_in_the_breeze,_spr_fps16_insert0-19-24-32_position0-5-7-9_lora0.3.mp4 /home/xinyu/projects/CogVideo/outputs/train/cogvideox-5b-i2v/multiple_frame_withval_multiple_videos/manual_select_10_iter10000/samples_dog_running/Rocket_man/A_golden_retriever,_sporting_sleek_black_sunglasses,_with_its_lengthy_fur_flowing_in_the_breeze,_spr_fps16_insert0-19-24-32_position0-5-7-9_lora0.3.mp4