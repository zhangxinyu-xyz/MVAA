import argparse
import os
import numpy as np
import imageio
from PIL import Image
from moviepy.editor import *
from tqdm import tqdm
import math

def gif_to_mp4(gif_path, output_path, resize_width=None, resize_height=None, fps=8):
    # 加载GIF文件
    clip = VideoFileClip(gif_path)
    
    # 如果指定了尺寸，则调整视频大小
    if resize_width is not None and resize_height is not None:
        clip = clip.resize(newsize=(resize_width, resize_height))
    
    # 将视频转换为MP4格式，并保存
    clip.write_videofile(output_path, codec='libx264', fps=fps)
    print(f"GIF 已保存为 {output_path}")

def mp4_to_gif(mp4_path, output_path, resize_width=None, resize_height=None, fps=8):
    # 加载MP4文件
    clip = VideoFileClip(mp4_path)
    
    # 如果指定了尺寸，则调整视频大小
    if resize_width is not None and resize_height is not None:
        clip = clip.resized(new_size=(resize_width, resize_height))
    
    # 将视频数据转换为帧
    video_data = [frame for frame in clip.iter_frames(fps=fps, dtype='uint8')]
    resized_video_data = [np.array(frame) for frame in video_data]
    
    # 保存为 GIF 文件
    frames_for_gif = [Image.fromarray(frame) for frame in resized_video_data]
    frames_for_gif[0].save(
        output_path, save_all=True, append_images=frames_for_gif[1:], loop=0, duration=int(1000 / fps)
    )
    
    print(f"视频已保存为 {output_path}")


def mp4_to_images(mp4_path, output_dir, fps=8):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载MP4文件
    clip = VideoFileClip(mp4_path)
    
    # 提取帧并保存为图片
    for i, frame in enumerate(clip.iter_frames(fps=fps, dtype='uint8')):
        frame_image = Image.fromarray(frame)
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        frame_image.save(frame_path)
    
    print(f"视频帧已保存到目录 {output_dir}")

def mp4_to_split_videos(mp4_path, output_dir, fps=8, num_video=4, num_row_video=5):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载MP4文件
    clip = VideoFileClip(mp4_path)
    # fps = clip.fps
    # 获取原视频的宽度和高度
    width, height = clip.size
    # 计算每个子视频的宽度和高度
    num_cols = num_row_video
    num_rows = int(math.ceil(num_video / num_row_video))
    single_width = int(width // num_cols)
    single_height = int(height // num_rows)
    # import pdb; pdb.set_trace()
    # 创建用于保存子视频片段的列表
    subclips = [[] for _ in range(num_video)]

    # 遍历视频帧，并将每个子图像添加到对应的子视频列表中
    for frame in clip.iter_frames(fps=fps, dtype='uint8'): 
        for row in range(num_rows):
            for col in range(num_cols):
                video_id = row * num_cols + col
                left = col * single_width
                right = left + single_width
                top = row * single_height
                bottom = top + single_height
                cropped_frame = frame[top:bottom, left:right]
                subclips[video_id].append(cropped_frame)
                if video_id == num_video - 1: break

    # 将每个子视频列表保存为一个新的MP4文件
    for video_id, frames in enumerate(subclips):
        output_path = os.path.join(output_dir, f"video_{video_id}.mp4")
        video_clip = ImageSequenceClip(frames, fps=fps)
        video_clip.write_videofile(output_path, codec='libx264', logger=None)
        
        frames_for_gif = [Image.fromarray(frame) for frame in frames]
        frames_for_gif[0].save(
            output_path.replace('.mp4', '.gif'), save_all=True, append_images=frames_for_gif[1:], loop=0, duration=int(1000 / fps)
        )

    print(f"子视频已保存到目录 {output_dir}")
        
def mp4_to_images_and_split(mp4_path, output_dir, fps=8, num_video=4, num_row_video=5):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # import pdb; pdb.set_trace()
    # 加载MP4文件
    clip = VideoFileClip(mp4_path)
    
    # 获取原图的宽度和高度
    width, height = clip.size
    # 计算每个子图像的宽度和高度
    num_cols = min(num_video, num_row_video)
    num_rows = int(math.ceil(num_video / num_row_video))
    single_width = int(width // num_cols)
    single_height = int(height // num_rows)
    # import pdb; pdb.set_trace()
    # 提取帧并保存为图片
    for i, frame in enumerate(clip.iter_frames(fps=fps, dtype='uint8')):
        frame_image = Image.fromarray(frame)
        for row in range(num_rows):
            for col in range(num_cols):
                video_id = row * num_cols + col
                if video_id >= num_video:
                    break
                left = col * single_width
                right = left + single_width
                top = row * single_height
                bottom = top + single_height
                cropped_image = frame_image.crop((left, top, right, bottom))
                
                frame_path = os.path.join(output_dir, f"video_{video_id}/frame_{i:04d}.jpg")
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                cropped_image.save(frame_path)
    
    print(f"视频帧已保存到目录 {output_dir}")

def gif_to_images(gif_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载GIF文件
    gif = Image.open(gif_path)
    frame_index = 0
    
    try:
        while True:
            frame_path = os.path.join(output_dir, f"frame_{frame_index:04d}.png")
            gif.save(frame_path)
            frame_index += 1
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    
    print(f"GIF 帧已保存到目录 {output_dir}")

def add_noise_to_image(image_path, output_path, noise_level=25):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 读取原始图像
    original_image = Image.open(image_path).convert('RGB')
    original_image = np.array(original_image)
    
    # 生成与原始图像大小相同的随机噪声
    noise = np.random.normal(0, noise_level, original_image.shape).astype('int')
    
    # 将噪声叠加到原始图像
    noisy_image = np.clip(original_image + noise, 0, 255).astype('uint8')
    
    # 保存添加噪声后的图像
    noisy_image_pil = Image.fromarray(noisy_image)
    noisy_image_pil.save(output_path)
    
    print(f"添加噪声的图像已保存为 {output_path}")

def change_fps(input_path, output_path, new_fps):
    # 加载视频
    clip = VideoFileClip(input_path)
    
    # 修改帧率（注意：这里仅更改输出时的帧率）
    clip = clip.set_fps(new_fps)
    
    # 写入输出视频
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=new_fps)

def reduce_to_target_frames(input_path, output_path, target_frame_count, fps=16):
    # 加载视频
    clip = VideoFileClip(input_path)

    # 当前帧总数 = 原始持续时间 × 帧率
    total_frames = int(clip.duration * fps)

    # 目标时长 = 目标帧数 / 帧率
    new_duration = target_frame_count / fps

    # 如果目标帧数大于现有帧数，就报错
    if target_frame_count > total_frames:
        raise ValueError(f"目标帧数 {target_frame_count} 多于原始帧数 {total_frames}，无法处理。")

    # 截取前 new_duration 秒的视频
    new_clip = clip.subclip(0, new_duration)

    # 设置输出帧率
    new_clip = new_clip.set_fps(fps)

    # 输出
    new_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=fps)

def merge_videos(video_paths, output_path='final_video.mp4', fps=16, save_audio=False):
    # 检查所有路径是否有效
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件未找到: {path}")

    # 加载视频剪辑
    clips = [VideoFileClip(path) for path in video_paths]
    
    # 拼接所有剪辑
    final_clip = concatenate_videoclips(clips, method='compose')

    if save_audio:
        # 获取第一个视频的音频
        first_audio = clips[0].audio
        # 设置第一个视频的音频为最终剪辑的音频
        final_clip = final_clip.set_audio(first_audio)
    
    # 写入输出文件
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=fps)
    print(f"拼接完成，保存为：{output_path}")
    

        
if __name__ == "__main__":
    # 使用argparse获取命令行参数
    parser = argparse.ArgumentParser(description="Convert a GIF to MP4 with optional resizing.")
    parser.add_argument("--input_path", type=str, nargs="+", help="Path to the input GIF file.")
    parser.add_argument("--output_path", type=str, default="", help="Path to save the output MP4 file.")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save the extracted frames.")
    parser.add_argument("--width", type=int, help="Width of the output video.", default=None)
    parser.add_argument("--height", type=int, help="Height of the output video.", default=None)
    parser.add_argument("-f", "--fps", type=int, default=8, help="FPS of output video")
    parser.add_argument("--type", type=str, help="Path to the input GIF file.", default='mp4_to_gif', choices=['gif_to_mp4', 'mp4_to_gif', 'mp4_to_images', 'gif_to_images', 'gif_to_images_noise', 'mp4_to_images_and_split', 'mp4_to_split_videos', 'change_fps', 'reduce_to_target_frames', 'merge_videos'])
    parser.add_argument("--noise_level", type=int, default=100, help="noise_level")
    parser.add_argument("--num_video", type=int, default=4, help="Split number of output video")
    parser.add_argument("--num_target_frames", type=int, default=4, help="Target number of output video frames")
    parser.add_argument("--save_audio", action="store_true", help="whether use the video's audio, if yes, use the first video's audio")
    # parser.add_argument("--select_file_path", type=str, default=None, help="Select Path to save the output GIF file.")
        
    args = parser.parse_args()
    
    # select_file_info = []
    # if args.select_file_path is not None:
    #     with open(args.select_file_path, 'r') as f:
    #         select_file_name = f.readlines()
    #     select_file_info = []
    #     for line in select_file_name:
    #         line = line.strip()
    #         prompt = line.split('.')[0]
    #         if 'peekaboo' in line: traj_id = int(line.split('_')[2])
    #         else: traj_id = int(line.split('_')[1])
    #         video_id = line.split('_')[3]
    #         select_file_info.append([prompt, traj_id, video_id])
            
    # 调用相应的功能
    if args.type == "mp4_to_gif":
        mp4_to_gif(args.input_path[0], args.output_path, args.width, args.height, fps=args.fps)
    elif args.type == "gif_to_mp4":
        gif_to_mp4(args.input_path[0], args.output_path)
    elif args.type == "mp4_to_images":
        mp4_to_images(args.input_path[0], args.output_dir, fps=args.fps)
    elif args.type == "gif_to_images":
        gif_to_images(args.input_path[0], args.output_dir)
    elif args.type == "gif_to_images_noise":
        if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) != 0:
            pass
        else:
            gif_to_images(args.input_path, args.output_dir)
        for img_name in [f for f in os.listdir(args.output_dir) if f.endswith('.png')]:
            add_noise_to_image(os.path.join(args.output_dir, img_name), os.path.join(args.output_dir, f'add_noise_{args.noise_level}', img_name), noise_level=args.noise_level)
    elif args.type == "mp4_to_images_and_split":
        mp4_to_images_and_split(args.input_path[0], args.output_dir, fps=args.fps, num_video=args.num_video)
    elif args.type == "mp4_to_split_videos":
        mp4_to_split_videos(args.input_path[0], args.output_dir, fps=args.fps, num_video=args.num_video)
    elif args.type == "change_fps":
        change_fps(args.input_path[0], args.output_path, args.fps)
    elif args.type == "reduce_to_target_frames":
        reduce_to_target_frames(args.input_path[0], args.output_path, args.num_target_frames, args.fps)
    elif args.type == "merge_videos":
        merge_videos(args.input_path, args.output_path, fps=args.fps, save_audio=args.save_audio)


#### examples
# python tools/convert_mp4_to_gif.py --input_path outputs/cogvideox-2b/car_turn_4s_fps8/baseline/A_lion_is_running_on_the_road/seed446_strength0.9_gs6.0_step50.mp4 --output_path outputs/cogvideox-2b/car_turn_4s_fps8/baseline/A_lion_is_running_on_the_road/seed446_strength0.9_gs6.0_step50.gif --width 384 --height 384 --type "mp4_to_gif"
# python tools/convert_mp4_to_gif.py --input_path outputs/cogvideox-2b/car_turn_4s_fps8/baseline/A_lion_is_running_on_the_road/seed446_strength0.9_gs6.0_step50.gif --output_dir outputs/cogvideox-2b/car_turn_4s_fps8/baseline/A_lion_is_running_on_the_road/seed446_strength0.9_gs6.0_step50 --type "gif_to_images"

# python tools/convert_mp4_to_gif.py --input_path outputs/cogvideox-2b/car_turn_4s_fps8/A_lion_is_running_on_the_road_nonorm/seed446_strength0.9_gs6.0_step50_mg0-50_ml10000000.0_mt-l2-sim-softmax_blocks-t15-t18-t21-t24-t27_sf0-1-2-3-4-5-6_gp_all_lrg8_recurstep1.mp4 --output_path outputs/cogvideox-2b/car_turn_4s_fps8/A_lion_is_running_on_the_road_nonorm/seed446_strength0.9_gs6.0_step50_mg0-50_ml10000000.0_mt-l2-sim-softmax_blocks-t15-t18-t21-t24-t27_sf0-1-2-3-4-5-6_gp_all_lrg8_recurstep1.gif --width 384 --height 384 --type "mp4_to_gif"
# python tools/convert_mp4_to_gif.py --input_path outputs/cogvideox-2b/car_turn_4s_fps8/A_lion_is_running_on_the_road_nonorm/seed446_strength0.9_gs6.0_step50_mg0-50_ml10000000.0_mt-l2-sim-softmax_blocks-t15-t18-t21-t24-t27_sf0-1-2-3-4-5-6_gp_all_lrg8_recurstep1.gif --output_dir outputs/cogvideox-2b/car_turn_4s_fps8/A_lion_is_running_on_the_road_nonorm/seed446_strength0.9_gs6.0_step50_mg0-50_ml10000000.0_mt-l2-sim-softmax_blocks-t15-t18-t21-t24-t27_sf0-1-2-3-4-5-6_gp_all_lrg8_recurstep1 --type "gif_to_images"

#### python convert_mp4_to_gif.py --input_path /data/xinyu/projects/CogVideo/resources/videos/Jingle_Bells_5s_36-41_christmas-cat-16fps_768x1360.mp4 --output_dir /data/xinyu/projects/CogVideo/resources/videos/Jingle_Bells_5s_36-41_christmas-cat-16fps_768x1360/ --width 1360 --height 768 --type "mp4_to_images" --fps 16
#### python convert_mp4_to_gif.py --input_path /data/xinyu/projects/CogVideo/resources/videos/Jingle_Bells_5s_36-41_christmas-cat-16fps_768x1360.mp4 --output_path /data/xinyu/projects/CogVideo/resources/videos/Jingle_Bells_5s_36-41_christmas-cat-8fps_768x1360.mp4 --fps 8 --type "change_fps"

########### merge videos
# python convert_mp4_to_gif.py --input_path assets/videos/Rocket_man/Couple_Walking.mp4 assets/videos/Rocket_man/Dog_running.mp4 assets/videos/Rocket_man/Dog_chief.mp4 --output_path assets/videos/Rocket_man/Rocket_man_concate.mp4 --fps 16 --save_audio --type "merge_videos"