import sys, os
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torchvision.transforms.functional import resize
import re
import copy

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")


##########  loaders  ##########


def load_prompts(prompt_path: Path) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) > 0]


def _strip_redundant_first_component(rel: Path, base: Path) -> Path:
    """
    If rel is like 'OpenViD_random1000/foo.mp4' and base ends with 'OpenViD_random1000',
    joining base / rel would duplicate the folder. Drop the first segment when it
    matches base.name (last component of data_root).
    """
    if not rel.parts:
        return rel
    if rel.parts[0] == base.name and len(rel.parts) > 1:
        return Path(*rel.parts[1:])
    return rel


def _line_relative_to_data_root(line: str, data_dir: Path) -> Path:
    """
    Path under data_root only: lines may omit 'data/' or repeat it; data_root supplies the prefix.
    Examples (data_root=data): 'OpenViD/x.mp4', 'data/OpenViD/x.mp4' -> both -> OpenViD/x.mp4.
    """
    rel = Path(line)
    data_dir = Path(data_dir)
    try:
        return rel.relative_to(data_dir)
    except ValueError:
        pass
    return _strip_redundant_first_component(rel, data_dir)


def load_videos(video_path: Path, data_dir: Path | None) -> List[Path]:
    """
    If data_dir is set, returns paths **relative to data_dir** only (no data_root prefix).
    Resolve on disk with: data_dir / path.
    """
    with open(video_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
    if data_dir is None:
        return [video_path.parent / line for line in lines]
    data_dir = Path(data_dir)
    return [_line_relative_to_data_root(line, data_dir) for line in lines]


def load_images(image_path: Path, data_dir: Path | None) -> List[Path]:
    with open(image_path, "r", encoding="utf-8") as file:
        return [Path(line.strip()) for line in file.readlines() if len(line.strip()) > 0]


def load_images_from_videos(videos_path: List[Path]) -> List[Path]:
    first_frames_dir = videos_path[0].parent.parent / "first_frames"
    first_frames_dir.mkdir(exist_ok=True)

    first_frame_paths = []
    for video_path in videos_path:
        frame_path = first_frames_dir / f"{video_path.stem}.png"
        if frame_path.exists():
            first_frame_paths.append(frame_path)
            continue

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read video: {video_path}")

        # Save frame as PNG with same name as video
        cv2.imwrite(str(frame_path), frame)
        logging.info(f"Saved first frame to {frame_path}")

        # Release video capture
        cap.release()

        first_frame_paths.append(frame_path)

    return first_frame_paths

def load_images_from_one_videos_multiple_frames(videos_path: List[Path]) -> List[Path]:
    all_frames_dir = Path(str(videos_path[0]).replace('.mp4', ''))
    all_frames_dir.mkdir(exist_ok=True)

    all_frame_paths = []
    for video_path in videos_path:
        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Read all frames
        frames = []

        while True:
            ret, frame = cap.read()
            # No read frame
            if not ret:
                break
            frames.append(frame)
            
        for frame_idx, frame in enumerate(frames):
            frame_path = all_frames_dir / f"frame_{frame_idx:04d}.png"
            if frame_path.exists():
                all_frame_paths.append(frame_path)
                continue
            else:
                # Save frame as PNG with same name as video
                cv2.imwrite(str(frame_path), frame)
                logging.info(f"Saved {frame_idx}-th frame to {frame_path}")
                all_frame_paths.append(frame_path)

        # Release video capture
        cap.release()
    return all_frame_paths

def load_images_from_multiple_videos_multiple_frames(videos_path: List[Path],
                                                max_num_frames: int,
                                                # height: int,
                                                # width: int,
                                                indices: list=None,) -> List[Path]:
    all_frames_dir = videos_path[0].parent
    all_frames_dir.mkdir(exist_ok=True)

    all_frame_paths = []
    for video_idx, video_path in enumerate(videos_path):
        single_frame_paths = []
        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Read all frames
        frames = []

        while True:
            ret, frame = cap.read()
            # No read frame
            if not ret:
                break
            frames.append(frame)
        
        video_num_frames = len(frames)
        if video_num_frames < max_num_frames:
            # Get all frames first
            frames = frames[:max_num_frames]
            # Repeat the last frame until we reach max_num_frames
            last_frame = frames[-1:]
            num_repeats = max_num_frames - video_num_frames
            repeated_frames = last_frame * num_repeats
            frames = frames + repeated_frames
        else:
            if indices is None:
                temp_indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
            else:
                temp_indices = indices
            frames = [frames[i] for i in temp_indices]
            frames = frames[:max_num_frames]
        
        save_path = all_frames_dir / f"{video_path.stem}"
        save_path.mkdir(exist_ok=True)
        for frame_idx, frame in enumerate(frames):
            frame_path = save_path / f"frame_{frame_idx:04d}.png"
            if frame_path.exists():
                single_frame_paths.append(frame_path)
                continue
            else:
                # Save frame as PNG with same name as video
                cv2.imwrite(str(frame_path), frame)
                logging.info(f"Saved {video_idx}-th video {frame_idx}-th frame to {frame_path}")
                single_frame_paths.append(frame_path)
         
        # Release video capture
        cap.release()
        all_frame_paths.append(single_frame_paths)
    return all_frame_paths

def numerical_sort_key(filename):
    # Extract digits from the string and convert them to an integer sort key.
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else -1

def load_images_from_folder(videos_path: List[Path]) -> List[Path]:
    assert os.path.exists(videos_path), f"{videos_path} is not exists."
    all_frame_paths = sorted(os.listdir(videos_path), key=numerical_sort_key)
    all_frame_paths = [Path(os.path.join(videos_path, _path)) for _path in all_frame_paths]
    return all_frame_paths

##########  preprocessors  ##########


def preprocess_image_with_resize(
    image_path: Path | str,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single image.

    Args:
        image_path: Path to the image file.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        torch.Tensor: Image tensor with shape [C, H, W] where:
            C = number of channels (3 for RGB)
            H = height
            W = width
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)
    image = cv2.imread(image_path.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1).contiguous()
    return image


def preprocess_video_with_resize(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
    indices: list=None,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count > max_num_frames, downsample frames evenly
      2. If video dimensions don't match (height, width), resize frames

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix(), width=width, height=height)
    video_num_frames = len(video_reader)
    if video_num_frames < max_num_frames:
        # Get all frames first
        frames = video_reader.get_batch(list(range(video_num_frames)))
        # Repeat the last frame until we reach max_num_frames
        last_frame = frames[-1:]
        num_repeats = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)
        return frames.float().permute(0, 3, 1, 2).contiguous()
    else:
        if indices is None:
            indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
        frames = video_reader.get_batch(indices)
        frames = frames[:max_num_frames].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        return frames


def preprocess_video_with_buckets(
    video_path: Path,
    resolution_buckets: List[Tuple[int, int, int]],
) -> torch.Tensor:
    """
    Args:
        video_path: Path to the video file.
        resolution_buckets: List of tuples (num_frames, height, width) representing
            available resolution buckets.

    Returns:
        torch.Tensor: Video tensor with shape [F, C, H, W] where:
            F = number of frames
            C = number of channels (3 for RGB)
            H = height
            W = width

    The function processes the video through these steps:
        1. Finds nearest frame bucket <= video frame count
        2. Downsamples frames evenly to match bucket size
        3. Finds nearest resolution bucket based on dimensions
        4. Resizes frames to match bucket resolution
    """
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    video_num_frames = len(video_reader)
    resolution_buckets = [bucket for bucket in resolution_buckets if bucket[0] <= video_num_frames]
    if len(resolution_buckets) == 0:
        raise ValueError(f"video frame count in {video_path} is less than all frame buckets {resolution_buckets}")

    nearest_frame_bucket = min(
        resolution_buckets,
        key=lambda bucket: video_num_frames - bucket[0],
        default=1,
    )[0]
    frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))
    frames = video_reader.get_batch(frame_indices)
    frames = frames[:nearest_frame_bucket].float()
    frames = frames.permute(0, 3, 1, 2).contiguous()

    nearest_res = min(resolution_buckets, key=lambda x: abs(x[1] - frames.shape[2]) + abs(x[2] - frames.shape[3]))
    nearest_res = (nearest_res[1], nearest_res[2])
    frames = torch.stack([resize(f, nearest_res) for f in frames], dim=0)

    return frames
