"""
Unified MVAA evaluation: Beat Alignment (BeatALign), Content Preservation (LPIPS),
and Temporal Consistency (clip_score_frame). Select metrics via --metrics.
"""
import os
import argparse
import random
import numpy as np
import torch
import cv2
import librosa
from scipy.signal import find_peaks
from glob import glob
from tqdm import tqdm
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
from transformers import AutoProcessor, CLIPModel


def set_seed(seed=42):
    """Fix random seeds and PyTorch/CUDA determinism for reproducible metrics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_txt_file(src_path):
    with open(src_path, "r") as f:
        return [line.strip() for line in f]


def get_generated_video_path(submission_path, video_name, music_name):
    p_plain = os.path.join(submission_path, video_name, music_name, "output.mp4")
    return p_plain

# ----- Beat Alignment -----
def compute_motion_intensity_per_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video: " + video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read first frame.")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_intensities = []
    frame_idx_offset = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray, prev_gray)
        motion_score = np.sum(frame_diff) / (255.0 * gray.shape[0] * gray.shape[1])
        motion_intensities.append(motion_score)
        prev_gray = gray
    cap.release()
    frame_indices = np.arange(frame_idx_offset, frame_idx_offset + len(motion_intensities))
    time_stamps = frame_indices / fps
    return {"frame_index": frame_indices, "time": time_stamps, "motion_intensity": np.array(motion_intensities), "fps": fps}


def extract_peaks(data, threshold=0.6, distance=5):
    if len(data["motion_intensity"]) == 0:
        return {"peak_indices": np.array([]), "peak_times": np.array([]), "peak_values": np.array([])}
    max_motion = np.max(data["motion_intensity"])
    norm_motion = data["motion_intensity"] / max_motion if max_motion > 0 else np.zeros_like(data["motion_intensity"])
    peaks, _ = find_peaks(norm_motion, height=threshold, distance=distance)
    return {"peak_indices": data["frame_index"][peaks], "peak_times": data["time"][peaks], "peak_values": data["motion_intensity"][peaks]}


def analyze_music_rhythm(audio_path, sr=22050):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print("Error loading audio:", audio_path, e)
        return np.array([]), 0, None, 0
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo.item()
    return beat_times, tempo, y, sr


def estimate_bas(music_beats, video_changes, sigma=0.05):
    if len(music_beats) == 0 or len(video_changes) == 0:
        return 0.0
    sigma_sq2 = 2 * (sigma ** 2)
    bas_score = sum(np.exp(-np.min((tv_i - music_beats) ** 2) / sigma_sq2) for tv_i in video_changes)
    return bas_score / len(video_changes)


def run_beat_align(videos, musics, audio_base, video_base, submission_path, sync_tolerance, peak_height=0.6, peak_dist=5):
    scores = []
    for video_path in tqdm(videos, desc="BeatALign"):
        video_name = os.path.basename(video_path).replace(".mp4", "")
        for music_path in musics:
            music_name = os.path.basename(music_path).replace(".mp3", "")
            music_full = os.path.join(audio_base, music_path)
            out_video = os.path.join(submission_path, video_name, music_name, "output.mp4")
            video_path_use = out_video
            music_beats, _, _, _ = analyze_music_rhythm(music_full)
            try:
                motion_data = compute_motion_intensity_per_frame(video_path_use)
                peak_info = extract_peaks(motion_data, threshold=peak_height, distance=peak_dist)
                video_changes = peak_info["peak_times"]
            except Exception as e:
                video_changes = np.array([])
            if len(music_beats) > 0 and len(video_changes) > 0:
                scores.append(estimate_bas(music_beats, video_changes, sigma=sync_tolerance))
            else:
                scores.append(0.0)
    return scores


# ----- LPIPS and clip_score_frame -----
def clip_score_frame(frames, processor, model, device):
    inputs = processor(images=frames, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs).detach().cpu().numpy()
    cosine_sim = cosine_similarity(image_features)
    np.fill_diagonal(cosine_sim, 0)
    return float(cosine_sim.sum() / (len(frames) * (len(frames) - 1)))


def lpips_video(ori_video, tgt_video, loss_fn, device):
    cap1 = cv2.VideoCapture(ori_video)
    cap2 = cv2.VideoCapture(tgt_video)
    frame_count = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
    lpips_scores = []
    for _ in range(frame_count):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        img1 = transform(frame1).unsqueeze(0).to(device)
        img2 = transform(frame2).unsqueeze(0).to(device)
        with torch.no_grad():
            lpips_scores.append(loss_fn(img1, img2).item())
    cap1.release()
    cap2.release()
    return np.mean([max(0.0, v) for v in lpips_scores]) if lpips_scores else 0.0


def run_lpips(videos, musics, video_base, submission_path, loss_fn, device):
    scores = []
    skipped = 0
    for video_path in tqdm(videos, desc="LPIPS"):
        video_name = os.path.basename(video_path).replace(".mp4", "")
        ori_video = os.path.join(video_base, video_path)
        if not os.path.exists(ori_video):
            skipped += len(musics)
            continue
        for music_path in musics:
            music_name = os.path.basename(music_path).replace(".mp3", "")
            tgt_video = get_generated_video_path(submission_path, video_name, music_name)
            if not os.path.exists(tgt_video):
                skipped += 1
                continue
            scores.append(lpips_video(ori_video, tgt_video, loss_fn, device))
    if skipped:
        print("LPIPS: skipped {} pairs (missing source or output.mp4).".format(skipped))
    return scores


def run_clip_score_frame(videos, musics, submission_path, model, processor, device, num_frames=48, w=480, h=480):
    scores = []
    for video_path in tqdm(videos, desc="clip_score_frame"):
        video_name = os.path.basename(video_path).replace(".mp4", "")
        for music_path in musics:
            music_name = os.path.basename(music_path).replace(".mp3", "")
            image_dir = os.path.join(submission_path, video_name, music_name, "output_frames")
            if not os.path.exists(image_dir):
                gen_video = get_generated_video_path(submission_path, video_name, music_name)
                if os.path.exists(gen_video):
                    cmd = "python preprocess/video_tools.py --type mp4_to_images --input_path " + gen_video + " --output_dir " + image_dir + " --fps 16"
                    os.system(cmd)
            if not os.path.exists(image_dir):
                continue
            paths = sorted(glob(os.path.join(image_dir, "*.png")))[:num_frames]
            if len(paths) < 2:
                continue
            frames = [Image.open(p).resize((w, h)) for p in paths]
            scores.append(clip_score_frame(frames, processor, model, device))
    return scores


METRICS_ALL = ["beat_align", "LPIPS", "clip_score_frame"]


def main():
    parser = argparse.ArgumentParser(description="MVAA evaluation: BeatALign, LPIPS, clip_score_frame.")
    parser.add_argument("--raw_input_txt", nargs=2, metavar=("MUSIC_TXT", "VIDEO_TXT"), required=True,
                        help="Paths to text files listing music and video paths.")
    parser.add_argument("--submission_path", type=str, required=True, help="Path to submission folder.")
    parser.add_argument("--metrics", nargs="+", choices=METRICS_ALL + ["all"], default=["all"],
                        help="Metrics to run: all, or one or more of beat_align, LPIPS, clip_score_frame.")
    parser.add_argument("--sync_tolerance", type=float, default=0.11, help="Sigma for Beat Alignment.")
    parser.add_argument("--num_frames", type=int, default=48, help="Frames for clip_score_frame.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_model_path", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--lpips_vgg_path", type=str, default=os.path.expanduser("~/.cache/torch/hub/checkpoints/vgg16-397923af.pth"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible LPIPS / clip_score_frame.")
    args = parser.parse_args()

    set_seed(args.seed)

    # Resolve all paths to absolute so results do not depend on current working directory
    music_txt = os.path.abspath(args.raw_input_txt[0])
    video_txt = os.path.abspath(args.raw_input_txt[1])
    submission_path = os.path.abspath(args.submission_path)
    audio_base = os.path.dirname(music_txt)
    video_base = os.path.dirname(video_txt)

    videos = load_txt_file(video_txt)
    musics = load_txt_file(music_txt)
    print("Videos:", len(videos), "Musics:", len(musics))
    print("Submission path:", submission_path)
    metrics = METRICS_ALL if "all" in args.metrics else args.metrics

    if "beat_align" in metrics:
        print("\n--- Beat Alignment (BeatALign) ---")
        bas_scores = run_beat_align(videos, musics, audio_base, video_base, submission_path, args.sync_tolerance)
        if bas_scores:
            print("BeatALign: mean = {:.4f}, n = {}".format(np.mean(bas_scores), len(bas_scores)))

    if "LPIPS" in metrics:
        print("\n--- Content Preservation (LPIPS) ---")
        import lpips
        loss_fn = lpips.LPIPS(net="vgg", model_path=args.lpips_vgg_path).to(args.device)
        lpips_scores = run_lpips(videos, musics, video_base, submission_path, loss_fn, args.device)
        if lpips_scores:
            print("LPIPS: mean = {:.4f}, n = {}".format(np.mean(lpips_scores), len(lpips_scores)))
        else:
            print("LPIPS: no pairs evaluated (check submission_path and that output.mp4 exist).")

    if "clip_score_frame" in metrics:
        print("\n--- Temporal Consistency (clip_score_frame) ---")
        model = CLIPModel.from_pretrained(args.clip_model_path).to(args.device)
        processor = AutoProcessor.from_pretrained(args.clip_model_path)
        tc_scores = run_clip_score_frame(videos, musics, submission_path, model, processor, args.device, args.num_frames)
        if tc_scores:
            print("clip_score_frame: mean = {:.4f}, n = {}".format(np.mean(tc_scores), len(tc_scores)))
    print("\nDone.")


if __name__ == "__main__":
    main()
