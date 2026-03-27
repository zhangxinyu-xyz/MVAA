import argparse
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import os
import glob

def merge_videos_with_audios(video_paths, audio_paths, output_path):
    if len(video_paths) != len(audio_paths):
        print("The number of video files does not match the number of audio files!")
        return

    final_clips = []

    for v_path, a_path in zip(video_paths, audio_paths):
        if not os.path.isfile(v_path):
            print(f"Video file does not exist, skipping: {v_path}")
            continue
        if not os.path.isfile(a_path):
            print(f"Audio file does not exist, skipping: {a_path}")
            continue

        video = VideoFileClip(v_path)
        audio = AudioFileClip(a_path)

        # Set the audio as the new audio track of the video
        video = video.set_audio(audio)

        final_clips.append(video)

    if not final_clips:
        print("No available video clips.")
        return

    final_video = concatenate_videoclips(final_clips, method="compose")
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")


def main():
    parser = argparse.ArgumentParser(description="Concatenate multiple videos with their corresponding audios.")
    parser.add_argument("--videos", nargs="+", required=True, help="List of video file paths")
    parser.add_argument("--audios", nargs="+", required=True, help="List of corresponding audio file paths")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("--start_time", type=int, required=False, help="Audio file start time")
    parser.add_argument("--end_time", type=int, required=False, help="Audio file end time")

    args = parser.parse_args()
    
    video_paths = args.videos
    audio_paths = args.audios
    print(f"Using command-line specified video files: {video_paths}")
    
    merge_videos_with_audios(video_paths, audio_paths, args.output)

if __name__ == "__main__":
    main()
