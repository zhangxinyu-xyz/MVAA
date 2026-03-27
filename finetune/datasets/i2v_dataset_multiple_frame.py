import os, sys, random, math
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override

from finetune.constants import LOG_LEVEL, LOG_NAME

from .utils import (
    load_images,
    load_images_from_videos,
    load_images_from_one_videos_multiple_frames,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
)


if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class BaseI2VDatasetMultipleFrame(Dataset):
    """
    Base dataset class for Image-to-Video (I2V) training.

    This dataset loads prompts, videos and corresponding conditioning images for I2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        image_column (str): Path to file containing image paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        data_root: str,
        caption_column: str,
        video_column: str,
        image_column: str | None,
        device: torch.device,
        trainer: "Trainer" = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        self.data_root = data_root
        self.prompts = load_prompts(data_root / caption_column)
        
        if video_column.suffix == ".txt":
            self.videos = load_videos(data_root / video_column, data_dir=data_root)
        elif video_column.suffix == ".mp4":
            try:
                self.videos = [(data_root / video_column).relative_to(data_root)]
            except ValueError:
                self.videos = [Path(video_column)]

        if image_column is not None and (data_root / image_column).exists():
            self.images = load_images(data_root / image_column, data_dir=data_root)
        else:
            self.images = load_images_from_one_videos_multiple_frames([data_root / v for v in self.videos])
            with (data_root / Path('images_' + str(video_column).replace('.mp4', ''))).open('w') as f:
                for item in self.images:
                    f.write(f"{item}\n")
                    
        self.frame_number = len(self.images)
        self.cur_frame_index_for_insert = None
        
        self.trainer = trainer
        
        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text

        # Check if number of prompts matches number of videos
        if not (len(self.videos) == len(self.prompts)):
            raise ValueError(
                f"Expected length of prompts, videos and images to be the same but found {len(self.prompts)=}, {len(self.videos)=}. Please ensure that the number of caption prompts, videos match in your dataset."
            )

        # Check if all video files exist
        if any(not (data_root / path).is_file() for path in self.videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(data_root / path for path in self.videos if not (data_root / path).is_file())}"
            )

        # Check if all image files exist
        if any(not path.is_file() for path in self.images):
            raise ValueError(
                f"Some image files were not found. Please ensure that all image files exist in the dataset directory. Missing file: {next(path for path in self.images if not path.is_file())}"
            )

    def __get_total_selection_number__(self, max_number=None):
        # total number of selection: C(frame_number,1)+C(frame_number,2)+...+C(frame_number,frame_number)=2^48 - 1
        if max_number is None: max_number = self.frame_number - 1
        total_selection_number = sum(math.comb(self.frame_number - 1, i) for i in range(1, max_number)) ## 2**48 - 1
        print("total selection number:", total_selection_number)
        self.total_selection_number = total_selection_number
        
    def __get_insert_index__(self):
        numbers = list(range(1, self.frame_number))
        
        # random select number k（range: 1 to frame_number - 1
        k = random.randint(1, min(len(numbers), 15))  #### TODO: in this stage, we only suppport 1 to 15 frames for anchors
        # print("number of select frames to insert except first frame:", k)

        # random select k number without repeat, ascend
        selected_numbers = sorted(random.sample(numbers, k))
        
        if self.trainer.args.data_select_maxframe is not None and len(selected_numbers) > self.trainer.args.data_select_maxframe - 1:
            selected_numbers = random.sample(selected_numbers, self.trainer.args.data_select_maxframe - 1)
        
        selected_numbers = [0] + selected_numbers
        # print("random selected frame index (ascend):", selected_numbers)

        return selected_numbers
    
    def __get_shift_insert_index__(self, ori_index, max_increment = 4):
        """
        Randomly add an offset to each number in an ascending list L such that
        new_L[i] = L[i] + δ[i],
        where the δ array is non-decreasing (ensuring new_L remains ascending),
        and new_L[-1] does not exceed maxframe.
        
        Parameters:
        L: An ascending list of numbers
        maxframe: The maximum value in the new list must not exceed maxframe
        max_increment: The upper bound of random offset at each step (each random increment is within [0, max_increment])
        
        Returns:
        new_L: The processed new list
        """

        if not ori_index:
            return []
        if ori_index[-1] > self.frame_number - 1:
            raise ValueError(f"The last frame index is {ori_index[-1]} large than the maxframe {self.frame_number}!")

        N = len(ori_index)
        # Total available offset to ensure the last number does not exceed maxframe
        total_available = self.frame_number - ori_index[-1]

        # Generate n random increments (each within [0, max_increment])
        increments = [random.uniform(0, max_increment) for _ in range(N)]
        
        # Compute cumulative offsets: δ[0] = increments[0], δ[i] = δ[i-1] + increments[i]
        cumulative = [0] * N
        cumulative[0] = increments[0]
        for i in range(1, N):
            cumulative[i] = cumulative[i-1] + increments[i]
        
        # If the cumulative sum exceeds total_available, scale it down proportionally
        if cumulative[-1] > total_available:
            scale = total_available / cumulative[-1]
            cumulative = [x * scale for x in cumulative]
        
        # Generate the new list: new_L[i] = L[i] + cumulative[i]
        new_L = [ori_index[i] + cumulative[i] for i in range(N)]
        new_L = list(map(int, new_L))
        
        # Check keep the first is 0, the last is less than self.frame_number
        new_L[0] = 0
        if new_L[-1] > self.frame_number - 1:
            new_L[-1] = self.frame_number - 1
        return new_L
        
    def __len__(self) -> int:
        ###### The max length of this dataset is set to self.frame_number, however, this can be set to others
        # return len(self.videos)
        return self.frame_number

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index
        
        selected_frame_index_for_insert = self.__get_insert_index__()
        
        prompt = self.prompts[0]
        video = self.trainer.args.data_root / self.videos[0]
        image = [self.images[select_index] for select_index in selected_frame_index_for_insert]
        
        #### whether use data shift for frames or not
        shifted_frame_index_for_insert = selected_frame_index_for_insert
        if self.trainer.args.data_shift and random.uniform(0, 1) <= self.trainer.args.data_shift_degree:
            shifted_frame_index_for_insert = self.__get_shift_insert_index__(selected_frame_index_for_insert)

        self.cur_frame_index_for_insert = shifted_frame_index_for_insert
        
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        cache_dir = self.trainer.args.data_root / "cache"
        video_latent_dir = cache_dir / "video_latent" / self.trainer.args.model_name / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        encoded_video_path = video_latent_dir / (video.stem + ".safetensors")

        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,
            )
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
            prompt_embedding = prompt_embedding[0]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)

        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            logger.debug(f"Loaded encoded video from {encoded_video_path}", main_process_only=False)
            # shape of image: [C, H, W]
            image = [self.preprocess(None, self.images[_index])[1] for _index in selected_frame_index_for_insert]
            image = [self.image_transform(_image) for _image in image]
        else:
            frames, image = self.preprocess(video, image) ### here image is a list, not a single file
            frames = frames.to(self.device)
            image = [_image.to(self.device) for _image in image]
            image = [self.image_transform(_image) for _image in image]
            # Current shape of frames: [F, C, H, W]
            frames = self.video_transform(frames)

            # Convert to [B, C, F, H, W]
            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            encoded_video = self.encode_video(frames)

            # [1, C, F, H, W] -> [C, F, H, W]
            encoded_video = encoded_video[0]
            encoded_video = encoded_video.to("cpu")
            image = [_image.to("cpu") for _image in image]
            save_file({"encoded_video": encoded_video}, encoded_video_path)
            logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)

        # shape of encoded_video: [C, F, H, W]
        # shape of image: [C, H, W]
        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
            "cur_frame_index_for_insert": self.cur_frame_index_for_insert,
        }

    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and preprocesses a video and an image.
        If either path is None, no preprocessing will be done for that input.

        Args:
            video_path: Path to the video file to load
            image_path: Path to the image file to load

        Returns:
            A tuple containing:
                - video(torch.Tensor) of shape [F, C, H, W] where F is number of frames,
                  C is number of channels, H is height and W is width
                - image(torch.Tensor) of shape [C, H, W]
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor
        """
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to an image.

        Args:
            image (torch.Tensor): A 3D tensor representing an image
                with shape [C, H, W] where:
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed image tensor
        """
        raise NotImplementedError("Subclass must implement this method")


class I2VDatasetMultipleFrameWithResize(BaseI2VDatasetMultipleFrame):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width)
        else:
            video = None
        if image_path is not None:
            if isinstance(image_path, Path):
                image = preprocess_image_with_resize(image_path, self.height, self.width)
            elif isinstance(image_path, list) or isinstance(image_path, tuple):
                image = [preprocess_image_with_resize(_path, self.height, self.width) for _path in image_path]
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)


class I2VDatasetWithBuckets(BaseI2VDatasetMultipleFrame):
    def __init__(
        self,
        video_resolution_buckets: List[Tuple[int, int, int]],
        vae_temporal_compression_ratio: int,
        vae_height_compression_ratio: int,
        vae_width_compression_ratio: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.video_resolution_buckets = [
            (
                int(b[0] / vae_temporal_compression_ratio),
                int(b[1] / vae_height_compression_ratio),
                int(b[2] / vae_width_compression_ratio),
            )
            for b in video_resolution_buckets
        ]
        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path, image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        video = preprocess_video_with_buckets(video_path, self.video_resolution_buckets)
        image = preprocess_image_with_resize(image_path, video.shape[2], video.shape[3])
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)
