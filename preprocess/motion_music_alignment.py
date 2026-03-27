import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import cv2, os, csv
from scipy.signal import find_peaks


def extract_audio_onset(audio_path, sr=None):
    y, sr = librosa.load(audio_path, sr=sr)
    # onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # times = librosa.times_like(onset_env, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return tempo, beat_times, beat_frames


def extract_motion_peaks(video_path, fps=16, threshold=0.6, distance=5):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(original_fps / fps))

    prev_gray = None
    motion = []
    time_stamps = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval != 0:
            idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion.append(np.sum(diff))
            time_stamps.append(idx / original_fps)
        prev_gray = gray
        idx += 1
    cap.release()

    motion = np.array(motion)
    motion = (motion - motion.min()) / (motion.max() - motion.min())  # normalize
    time_stamps = np.array(time_stamps)
    peak_indices, _ = find_peaks(motion, height=threshold, distance=distance)
    motion_peaks = time_stamps[peak_indices]
    return time_stamps, motion, motion_peaks

def match_beats_to_motion(music_beats, motion_peaks):
    """
    Matches each music beat to the closest unused motion peak in time order.
    Ensures one-to-one matching without repetition.
    """
    matched = []
    used_indices = set()

    for b in music_beats:
        min_dist = float("inf")
        chosen_idx = -1
        for i, mp in enumerate(motion_peaks):
            if i in used_indices:
                continue
            dist = abs(b - mp)
            if dist < min_dist:
                min_dist = dist
                chosen_idx = i
        if chosen_idx != -1:
            used_indices.add(chosen_idx)
            matched.append((b, motion_peaks[chosen_idx]))
    return matched

def save_matched_pairs(data_list, output_csv_filename):
    # Define CSV title
    header = ['beat_time', 'peak_time']

    try:
        with open(output_csv_filename, 'w', newline='') as csvfile:
            # Create a new csv writer object
            csv_writer = csv.writer(csvfile)
            # Write title
            csv_writer.writerow(header)
            # Write data line
            csv_writer.writerows(data_list)

        print(f"Data has been saved to: {output_csv_filename}")

    except IOError as e:
        print(f"Error in writing: {e}")
    except Exception as e:
        print(f"Unknow error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Visualize alignment between music beats and video motion.")
    parser.add_argument("--video", required=True, help="Path to input video (e.g., .mp4)")
    parser.add_argument("--audio", required=True, help="Path to input audio (e.g., .mp3)")
    parser.add_argument("--output", required=True, help="Output path for motion_beat alignment")

    args = parser.parse_args()

    print("🎵 Step 1: Extracting music beat envelope...")
    time, music_beats, beat_frames = extract_audio_onset(args.audio)

    print("🎬 Step 2: Extracting motion peaks from video...")
    _, _, motion_peaks = extract_motion_peaks(args.video)

    print("🔗 Step 3: Maching motion peaks with music beats...")
    matched_pairs = match_beats_to_motion(music_beats, motion_peaks)
    print("Matched Beats to Motion Peaks:")
    for i, (beat, motion) in enumerate(matched_pairs):
        print(f"  Beat {i+1}: {beat:.2f}s -> Motion Peak at {motion:.2f}s")

    save_csv_name = args.output.replace(".mp4", "").replace(".mp3", ".csv")
    os.makedirs(os.path.dirname(save_csv_name), exist_ok=True)
    save_matched_pairs(matched_pairs, save_csv_name)


if __name__ == "__main__":
    main()
