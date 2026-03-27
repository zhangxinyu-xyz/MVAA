import argparse
import os
from PIL import Image
from moviepy.editor import *

def mp4_to_images(mp4_path, output_dir, fps=8):
    os.makedirs(output_dir, exist_ok=True)
    
    clip = VideoFileClip(mp4_path)
    
    for i, frame in enumerate(clip.iter_frames(fps=fps, dtype='uint8')):
        frame_image = Image.fromarray(frame)
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        frame_image.save(frame_path)
    
    print(f"Save frames to: {output_dir}")
        
if __name__ == "__main__":
    # Use argparse get hyperparameters from command line.
    parser = argparse.ArgumentParser(description="Convert a GIF to MP4 with optional resizing.")
    parser.add_argument("--input_path", type=str, nargs="+", help="Path to the input GIF file.")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save the extracted frames.")
    parser.add_argument("-f", "--fps", type=int, default=16, help="FPS of output video")
    parser.add_argument("--type", type=str, help="Path to the input GIF file.", default='mp4_to_images', choices=['mp4_to_images'])

    args = parser.parse_args()
    
    if args.type == "mp4_to_images":
        mp4_to_images(args.input_path[0], args.output_dir, fps=args.fps)
    else:
        raise ValueError(f"Invalid type: {args.type}")