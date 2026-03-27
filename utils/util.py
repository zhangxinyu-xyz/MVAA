import os
import pandas as pd

def load_music_beats_to_video_motion(music_beats_to_video_motion_path):
    df = pd.read_csv(music_beats_to_video_motion_path)
    # print(df)
    beat_frames = df["beat_time"].to_list()
    beat_frames = [round((round((_beat_frame + 1e-6) * 16) + 1e-6) / 4) for _beat_frame in beat_frames]
    beat_frames = [0] + beat_frames
    motion_frames = df["peak_time"].to_list()
    motion_frames = [round((_motion_frame + 1e-6)  * 16) for _motion_frame in motion_frames]
    motion_frames = [0] + motion_frames
    return motion_frames, beat_frames


