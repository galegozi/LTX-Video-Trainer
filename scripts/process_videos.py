#!/usr/bin/env python3

"""
Compute latent representations for video generation training.

This module provides functionality for processing video and image files, including:
- Loading videos/images from various file formats (CSV, JSON, JSONL)
- Resizing, cropping, and transforming media
- MediaDataset for video-only preprocessing workflows
- BucketSampler for grouping videos by resolution

Can be used as a standalone script:
    python -m ltxv_trainer.process_videos dataset.csv --resolution-buckets 768x768x25 --output-dir /path/to/output
"""

import json  # noqa: I001
from pathlib import Path
from typing import Any


import numpy as np
import pandas as pd
import torch
import typer
from pillow_heif import register_heif_opener
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import crop, resize, to_tensor
from transformers.utils.logging import disable_progress_bar

from ltxv_trainer import logger
from ltxv_trainer.ltxv_utils import encode_video
from ltxv_trainer.model_loader import LtxvModelVersion, load_vae
from ltxv_trainer.utils import open_image_as_srgb

# Should be imported after `torch` to avoid compatibility issues.
import decord  # type: ignore

decord.bridge.set_bridge("torch")
disable_progress_bar()

# Register HEIF/HEIC support
register_heif_opener()

# Constants for validation
VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Process videos/images and save latent representations for video generation training.",
)


class MediaDataset(Dataset):
    """
    Dataset for processing video and image files.

    This dataset is designed for media preprocessing workflows where you need to:
    - Load and preprocess videos/images
    - Apply resizing and cropping transformations
    - Handle different resolution buckets
    - Filter out invalid media files
    """

    def __init__(
        self,
        dataset_file: str | Path,
        main_media_column: str,
        video_column: str,
        resolution_buckets: list[tuple[int, int, int]],
        reshape_mode: str = "center",
    ) -> None:
        """
        Initialize the media dataset.

        Args:
            dataset_file: Path to CSV/JSON/JSONL metadata file
            video_column: Column name for video paths in the metadata file
            resolution_buckets: List of (frames, height, width) tuples
            reshape_mode: How to crop videos ("center", "random")
        """
        super().__init__()

        self.dataset_file = Path(dataset_file)
        self.main_media_column = main_media_column
        self.resolution_buckets = resolution_buckets
        self.reshape_mode = reshape_mode

        # First load main media paths
        self.main_media_paths = self._load_video_paths(main_media_column)

        # Then load reference video paths
        self.video_paths = self._load_video_paths(video_column)

        # Filter out videos with insufficient frames
        self._filter_valid_videos()

        self.max_num_frames = max(self.resolution_buckets, key=lambda x: x[0])[0]

        # Set up video transforms
        self.transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.clamp_(0, 1)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single video/image with metadata."""
        if isinstance(index, list):
            # Special case for BucketSampler - return cached data
            return index

        video_path: Path = self.video_paths[index]

        # Compute relative path of the video
        data_root = self.dataset_file.parent
        relative_path = str(video_path.relative_to(data_root))
        media_relative_path = str(self.main_media_paths[index].relative_to(data_root))

        if video_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            video = self._preprocess_image(video_path)
            fps = 1
        elif video_path.suffix.lower() == ".npz":
            # Check if this is a single NPZ file or a directory of NPZ files
            # If the path points to a directory, treat it as a sequence
            if video_path.is_dir():
                video, fps = self._preprocess_npz_sequence(video_path)
            else:
                video, fps = self._preprocess_npz(video_path)
        else:
            video, fps = self._preprocess_video(video_path)

        return {
            "video": video,
            "relative_path": relative_path,
            "main_media_relative_path": media_relative_path,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
                "fps": fps,
            },
        }

    def _load_video_paths(self, column: str) -> list[Path]:
        """Load video paths from the specified data source."""
        if self.dataset_file.suffix == ".csv":
            return self._load_video_paths_from_csv(column)
        elif self.dataset_file.suffix == ".json":
            return self._load_video_paths_from_json(column)
        elif self.dataset_file.suffix == ".jsonl":
            return self._load_video_paths_from_jsonl(column)
        else:
            raise ValueError("Expected `dataset_file` to be a path to a CSV, JSON, or JSONL file.")

    def _load_video_paths_from_csv(self, column: str) -> list[Path]:
        """Load video paths from a CSV file."""
        df = pd.read_csv(self.dataset_file)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV file")

        data_root = self.dataset_file.parent
        video_paths = [data_root / Path(line.strip()) for line in df[column].tolist()]

        # Validate that all paths exist (can be files or directories for NPZ sequences)
        invalid_paths = [path for path in video_paths if not (path.is_file() or path.is_dir())]
        if invalid_paths:
            raise ValueError(f"Found {len(invalid_paths)} invalid paths. First few: {invalid_paths[:5]}")

        return video_paths

    def _load_video_paths_from_json(self, column: str) -> list[Path]:
        """Load video paths from a JSON file."""
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")

        data_root = self.dataset_file.parent
        video_paths = []
        for entry in data:
            if column not in entry:
                raise ValueError(f"Key '{column}' not found in JSON entry")
            video_paths.append(data_root / Path(entry[column].strip()))

        # Validate that all paths exist (can be files or directories for NPZ sequences)
        invalid_paths = [path for path in video_paths if not (path.is_file() or path.is_dir())]
        if invalid_paths:
            raise ValueError(f"Found {len(invalid_paths)} invalid paths. First few: {invalid_paths[:5]}")

        return video_paths

    def _load_video_paths_from_jsonl(self, column: str) -> list[Path]:
        """Load video paths from a JSONL file."""
        data_root = self.dataset_file.parent
        video_paths = []
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                if column not in entry:
                    raise ValueError(f"Key '{column}' not found in JSONL entry")
                video_paths.append(data_root / Path(entry[column].strip()))

        # Validate that all paths exist (can be files or directories for NPZ sequences)
        invalid_paths = [path for path in video_paths if not (path.is_file() or path.is_dir())]
        if invalid_paths:
            raise ValueError(f"Found {len(invalid_paths)} invalid paths. First few: {invalid_paths[:5]}")

        return video_paths

    def _filter_valid_videos(self) -> None:
        """Filter out videos with insufficient frames."""
        original_length = len(self.video_paths)
        valid_video_paths = []
        min_frames_required = min(self.resolution_buckets, key=lambda x: x[0])[0]

        for video_path in self.video_paths:
            if video_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                valid_video_paths.append(video_path)
                continue
            
            if video_path.suffix.lower() == ".npz" or (video_path.is_dir() and list(video_path.glob("*.npz"))):
                # Check NPZ file or NPZ sequence for sufficient frames
                try:
                    if video_path.is_dir():
                        # NPZ sequence - count number of NPZ files
                        npz_files = list(video_path.glob("*.npz"))
                        num_frames = len(npz_files)
                        if num_frames >= min_frames_required:
                            valid_video_paths.append(video_path)
                        else:
                            logger.warning(
                                f"Skipping NPZ sequence at {video_path} - has {num_frames} frames, "
                                f"which is less than the minimum required frames ({min_frames_required})"
                            )
                    else:
                        # Single NPZ file
                        data = np.load(video_path)
                        
                        # Check if it has 'frames' or 'data' key (multi-frame format)
                        if 'frames' in data or 'data' in data:
                            frames_key = 'frames' if 'frames' in data else 'data'
                            frames = data[frames_key]
                            num_frames = frames.shape[0]
                            
                            if num_frames >= min_frames_required:
                                valid_video_paths.append(video_path)
                            else:
                                logger.warning(
                                    f"Skipping NPZ file at {video_path} - has {num_frames} frames, "
                                    f"which is less than the minimum required frames ({min_frames_required})"
                                )
                        else:
                            # Single frame with channels - treat as single frame
                            if min_frames_required <= 1:
                                valid_video_paths.append(video_path)
                            else:
                                logger.warning(
                                    f"Skipping NPZ file at {video_path} - contains single frame with channels, "
                                    f"but minimum required frames is {min_frames_required}"
                                )
                except Exception as e:
                    logger.warning(f"Failed to read NPZ at {video_path}: {e!s}")
                continue

            try:
                video_reader = decord.VideoReader(uri=video_path.as_posix())
                if len(video_reader) >= min_frames_required:
                    valid_video_paths.append(video_path)
                else:
                    logger.warning(
                        f"Skipping video at {video_path} - has {len(video_reader)} frames, "
                        f"which is less than the minimum required frames ({min_frames_required})"
                    )
            except Exception as e:
                logger.warning(f"Failed to read video at {video_path}: {e!s}")

        # Update video paths to only include valid ones
        self.video_paths = valid_video_paths

        if len(self.video_paths) < original_length:
            logger.warning(
                f"Filtered out {original_length - len(self.video_paths)} videos with insufficient frames. "
                f"Proceeding with {len(self.video_paths)} valid videos."
            )

    def _preprocess_image(self, path: Path) -> torch.Tensor:
        """Preprocess a single image by resizing and applying transforms."""
        image = open_image_as_srgb(path)
        image = to_tensor(image)
        image = image.unsqueeze(0)  # Add batch dimension to match video format

        # Find nearest resolution bucket and resize
        nearest_res = self._find_nearest_resolution(image.shape[2], image.shape[3])
        image_resized = self._resize_and_crop(image, nearest_res)

        # Apply transforms and ensure single frame
        image = self.transforms(image_resized)
        image = image.unsqueeze(0)  # Add frame dimension [1,C,H,W]
        return image

    def _preprocess_video(self, path: Path) -> tuple[torch.Tensor, float]:
        """Preprocess a video by loading, resizing, and applying transforms."""
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)
        fps = video_reader.get_avg_fps()

        relevant_buckets = [bucket for bucket in self.resolution_buckets if bucket[0] <= video_num_frames]
        nearest_frame_bucket = min(
            relevant_buckets,
            key=lambda x: abs(x[0] - min(video_num_frames, self.max_num_frames)),
            default=[1],
        )[0]

        frame_indices = list(range(video_num_frames))
        frames = video_reader.get_batch(frame_indices)
        if isinstance(frames, decord.ndarray.NDArray):
            frames = torch.from_numpy(frames.asnumpy())
        frames = frames[:nearest_frame_bucket].float() / 255.0
        frames = frames.permute(0, 3, 1, 2).contiguous()

        nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames_resized = self._resize_and_crop(frames, nearest_res)
        frames = torch.stack([self.transforms(frame) for frame in frames_resized], dim=0)

        return frames, fps

    def _preprocess_npz(self, path: Path) -> tuple[torch.Tensor, float]:
        """Preprocess an NPZ file containing physics simulation frames.
        
        Supports two formats:
        1. Single NPZ with all frames: 'frames' key with shape (T, H, W, C) or (T, C, H, W)
        2. Single NPZ with multiple channels: Multiple keys, each representing a channel
        
        Expected NPZ format:
        - Format 1: 'frames' or 'data': numpy array of shape (T, H, W, C) or (T, C, H, W)
        - Format 2: Multiple channel keys (e.g., 'temperature', 'pressure', 'velocity_x')
                    Each channel should be a 2D array (H, W) for single frame
        - 'fps': (optional) frames per second, defaults to 24
        - 'channels': (optional) list of channel names to use, in order
        
        The frames array should contain normalized values in range [0, 1].
        If values are in [0, 255], they will be automatically normalized.
        
        Args:
            path: Path to the NPZ file
            
        Returns:
            Tuple of (frames tensor, fps)
        """
        # Load NPZ file
        data = np.load(path)
        
        # Get FPS if available, otherwise default to 24
        fps = float(data.get('fps', 24))
        
        # Check if this is Format 1 (frames key) or Format 2 (multiple channels)
        if 'frames' in data or 'data' in data:
            # Format 1: Single array with all frames
            frames = data['frames'] if 'frames' in data else data['data']
            frames = torch.from_numpy(frames).float()
            
            # Handle different input formats
            # If frames are in format (T, H, W, C), convert to (T, C, H, W)
            if frames.ndim == 4 and frames.shape[-1] in [1, 3, 4]:
                frames = frames.permute(0, 3, 1, 2)
            elif frames.ndim == 3:
                # Grayscale (T, H, W) -> (T, 1, H, W)
                frames = frames.unsqueeze(1)
            elif frames.ndim != 4:
                raise ValueError(
                    f"Expected frames to have 3 or 4 dimensions, got {frames.ndim}. "
                    f"Shape: {frames.shape}"
                )
        else:
            # Format 2: Multiple channel keys (single frame)
            # Get channel names from 'channels' key or use first 3 available keys
            available_keys = [k for k in data.keys() if k not in ['fps', 'channels']]
            
            if 'channels' in data:
                # Use specified channels
                channel_names = data['channels']
                if isinstance(channel_names, np.ndarray):
                    channel_names = channel_names.tolist()
                    if isinstance(channel_names[0], bytes):
                        channel_names = [c.decode('utf-8') for c in channel_names]
            else:
                # Use first available channels (up to 3 for RGB)
                channel_names = available_keys[:3]
            
            if not channel_names:
                raise ValueError(
                    f"No valid channel data found in NPZ file. "
                    f"Available keys: {list(data.keys())}"
                )
            
            # Load and stack channels
            channel_data = []
            for channel_name in channel_names:
                if channel_name not in data:
                    raise ValueError(
                        f"Channel '{channel_name}' not found in NPZ file. "
                        f"Available keys: {list(data.keys())}"
                    )
                channel = data[channel_name]
                
                # Ensure channel is 2D (H, W)
                if channel.ndim == 2:
                    channel_data.append(torch.from_numpy(channel).float())
                else:
                    raise ValueError(
                        f"Expected channel '{channel_name}' to be 2D (H, W), "
                        f"got shape {channel.shape}"
                    )
            
            # Stack channels: (C, H, W)
            frames = torch.stack(channel_data, dim=0)
            # Add time dimension: (1, C, H, W)
            frames = frames.unsqueeze(0)
        
        # Normalize to [0, 1] if values are in [0, 255] or larger
        if frames.max() > 1.0:
            if frames.max() <= 255.0:
                frames = frames / 255.0
            else:
                # Normalize to [0, 1] based on actual range
                min_val = frames.min()
                max_val = frames.max()
                frames = (frames - min_val) / (max_val - min_val + 1e-8)
        
        # Handle channel count
        num_channels = frames.shape[1]
        
        if num_channels == 4:
            # Drop alpha channel, keep RGB
            frames = frames[:, :3, :, :]
        elif num_channels == 1:
            # Grayscale - expand to 3 channels
            frames = frames.repeat(1, 3, 1, 1)
        elif num_channels == 2:
            # Two channels - add a zero channel to make RGB
            zero_channel = torch.zeros_like(frames[:, :1, :, :])
            frames = torch.cat([frames, zero_channel], dim=1)
        elif num_channels > 4:
            # More than 4 channels - take first 3
            logger.warning(
                f"NPZ file has {num_channels} channels. Using only the first 3 for RGB representation."
            )
            frames = frames[:, :3, :, :]
        elif num_channels != 3:
            raise ValueError(
                f"Unexpected number of channels: {num_channels}. "
                f"Expected 1, 2, 3, or 4 channels."
            )
        
        video_num_frames = frames.shape[0]
        
        # Select appropriate frame bucket
        relevant_buckets = [bucket for bucket in self.resolution_buckets if bucket[0] <= video_num_frames]
        nearest_frame_bucket = min(
            relevant_buckets,
            key=lambda x: abs(x[0] - min(video_num_frames, self.max_num_frames)),
            default=[1],
        )[0]
        
        # Take only the required number of frames
        frames = frames[:nearest_frame_bucket]
        
        # Resize and crop to target resolution
        nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames_resized = self._resize_and_crop(frames, nearest_res)
        
        # Apply transforms
        frames = torch.stack([self.transforms(frame) for frame in frames_resized], dim=0)
        
        return frames, fps

    def _preprocess_npz_sequence(self, path: Path) -> tuple[torch.Tensor, float]:
        """Preprocess a sequence of NPZ files (one per frame).
        
        Each NPZ file should contain:
        - Multiple channel keys (e.g., 'temperature', 'pressure', 'velocity_x')
        - Each channel is a 2D array (H, W)
        - Optional 'channels' key specifying which channels to use and in what order
        - Optional 'fps' key (only read from first file)
        
        Args:
            path: Path to directory containing NPZ files
            
        Returns:
            Tuple of (frames tensor, fps)
        """
        # Get all NPZ files in directory, sorted by name
        npz_files = sorted(path.glob("*.npz"))
        
        if not npz_files:
            raise ValueError(f"No NPZ files found in directory: {path}")
        
        # Load first file to get FPS and channel configuration
        first_data = np.load(npz_files[0])
        fps = float(first_data.get('fps', 24))
        
        # Get channel names
        available_keys = [k for k in first_data.keys() if k not in ['fps', 'channels']]
        
        if 'channels' in first_data:
            channel_names = first_data['channels']
            if isinstance(channel_names, np.ndarray):
                channel_names = channel_names.tolist()
                if isinstance(channel_names[0], bytes):
                    channel_names = [c.decode('utf-8') for c in channel_names]
        else:
            # Use first 3 available channels for RGB
            channel_names = available_keys[:3]
        
        if not channel_names:
            raise ValueError(
                f"No valid channel data found in NPZ files. "
                f"Available keys in first file: {list(first_data.keys())}"
            )
        
        logger.info(f"Loading NPZ sequence from {path.name} with {len(npz_files)} frames using channels: {channel_names}")
        
        # Determine how many frames to load based on resolution buckets
        max_frames_needed = max(bucket[0] for bucket in self.resolution_buckets)
        frames_to_load = min(len(npz_files), max_frames_needed)
        
        # Load frames
        all_frames = []
        for npz_file in npz_files[:frames_to_load]:
            frame_data = np.load(npz_file)
            
            # Load and stack channels for this frame
            channel_data = []
            for channel_name in channel_names:
                if channel_name not in frame_data:
                    raise ValueError(
                        f"Channel '{channel_name}' not found in {npz_file.name}. "
                        f"Available keys: {list(frame_data.keys())}"
                    )
                channel = frame_data[channel_name]
                
                if channel.ndim != 2:
                    raise ValueError(
                        f"Expected channel '{channel_name}' in {npz_file.name} to be 2D (H, W), "
                        f"got shape {channel.shape}"
                    )
                
                channel_data.append(torch.from_numpy(channel).float())
            
            # Stack channels: (C, H, W)
            frame = torch.stack(channel_data, dim=0)
            all_frames.append(frame)
        
        # Stack all frames: (T, C, H, W)
        frames = torch.stack(all_frames, dim=0)
        
        # Normalize to [0, 1] if values are in [0, 255] or larger
        if frames.max() > 1.0:
            if frames.max() <= 255.0:
                frames = frames / 255.0
            else:
                # Normalize to [0, 1] based on actual range
                min_val = frames.min()
                max_val = frames.max()
                frames = (frames - min_val) / (max_val - min_val + 1e-8)
        
        # Handle channel count
        num_channels = frames.shape[1]
        
        if num_channels == 4:
            frames = frames[:, :3, :, :]
        elif num_channels == 1:
            frames = frames.repeat(1, 3, 1, 1)
        elif num_channels == 2:
            zero_channel = torch.zeros_like(frames[:, :1, :, :])
            frames = torch.cat([frames, zero_channel], dim=1)
        elif num_channels > 4:
            logger.warning(
                f"NPZ sequence has {num_channels} channels. Using only the first 3 for RGB representation."
            )
            frames = frames[:, :3, :, :]
        elif num_channels != 3:
            raise ValueError(f"Unexpected number of channels: {num_channels}")
        
        video_num_frames = frames.shape[0]
        
        # Select appropriate frame bucket
        relevant_buckets = [bucket for bucket in self.resolution_buckets if bucket[0] <= video_num_frames]
        nearest_frame_bucket = min(
            relevant_buckets,
            key=lambda x: abs(x[0] - min(video_num_frames, self.max_num_frames)),
            default=[1],
        )[0]
        
        # Take only the required number of frames
        frames = frames[:nearest_frame_bucket]
        
        # Resize and crop to target resolution
        nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
        frames_resized = self._resize_and_crop(frames, nearest_res)
        
        # Apply transforms
        frames = torch.stack([self.transforms(frame) for frame in frames_resized], dim=0)
        
        return frames, fps

    def _find_nearest_resolution(self, height: int, width: int) -> tuple[int, int]:
        """Find the nearest resolution bucket for the given dimensions."""
        nearest_res = min(self.resolution_buckets, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]

    def _resize_and_crop(self, arr: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """Resize and crop tensor to target size."""
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if self.reshape_mode == "random":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif self.reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise ValueError(f"Unsupported reshape mode: {self.reshape_mode}")

        arr = crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr


def compute_video_latents(
    dataset_file: str | Path,
    video_column: str,
    resolution_buckets: list[tuple[int, int, int]],
    output_dir: str,
    model_source: str,
    main_media_column: str | None = None,
    reshape_mode: str = "center",
    batch_size: int = 1,
    device: str = "cuda",
    vae_tiling: bool = False,
) -> None:
    """
    Process videos and save latent representations.

    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL) containing video paths
        video_column: Column name for video paths in the metadata file
        resolution_buckets: List of (frames, height, width) tuples
        output_dir: Directory to save latents
        model_source: Model source for VAE
        reshape_mode: How to crop videos ("center", "random")
        main_media_column: Column name for main media paths (if different from video_column)
        batch_size: Batch size for processing
        device: Device to use for computation
        vae_tiling: Whether to enable VAE tiling
    """
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from torch.utils.data import DataLoader

    console = Console()

    # Create dataset
    dataset = MediaDataset(
        dataset_file=dataset_file,
        main_media_column=main_media_column or video_column,
        video_column=video_column,
        resolution_buckets=resolution_buckets,
        reshape_mode=reshape_mode,
    )
    logger.info(f"Loaded {len(dataset)} valid media files")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load VAE
    with console.status(f"[bold]Loading VAE from [cyan]{model_source}[/]...", spinner="dots"):
        vae = load_vae(model_source, dtype=torch.bfloat16).to(device)

    if vae_tiling:
        vae.enable_tiling()

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Process batches
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing videos", total=len(dataloader))

        for batch in dataloader:
            # Encode videos
            with torch.inference_mode():
                video_latents = encode_video(
                    vae=vae,
                    image_or_video=batch["video"],
                    device=device,
                )

            # Save latents for each item in batch
            for i in range(len(batch["relative_path"])):
                output_rel_path = Path(batch["main_media_relative_path"][i]).with_suffix(".pt")
                output_file = output_path / output_rel_path

                # Create output directory maintaining structure
                output_file.parent.mkdir(parents=True, exist_ok=True)

                latent_data = {
                    "latents": video_latents["latents"][i].cpu().contiguous(),
                    "num_frames": video_latents["num_frames"],
                    "height": video_latents["height"],
                    "width": video_latents["width"],
                    "fps": batch["video_metadata"]["fps"][i].item(),
                }

                torch.save(latent_data, output_file)

            progress.advance(task)

    logger.info(f"Processed {len(dataset)} videos. Latents saved to {output_path}")


def parse_resolution_buckets(resolution_buckets_str: str) -> list[tuple[int, int, int]]:
    """Parse resolution buckets from string format to list of tuples (frames, height, width)"""
    resolution_buckets = []
    for bucket_str in resolution_buckets_str.split(";"):
        w, h, f = map(int, bucket_str.split("x"))

        if w % VAE_SPATIAL_FACTOR != 0 or h % VAE_SPATIAL_FACTOR != 0:
            raise typer.BadParameter(
                f"Width and height must be multiples of {VAE_SPATIAL_FACTOR}, got {w}x{h}",
                param_hint="resolution-buckets",
            )

        if f % VAE_TEMPORAL_FACTOR != 1:
            raise typer.BadParameter(
                f"Number of frames must be a multiple of {VAE_TEMPORAL_FACTOR} plus 1, got {f}",
                param_hint="resolution-buckets",
            )

        resolution_buckets.append((f, h, w))
    return resolution_buckets


@app.command()
def main(
    dataset_file: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) containing video paths",
    ),
    resolution_buckets: str = typer.Option(
        ...,
        help='Resolution buckets in format "WxHxF;WxHxF;..." (e.g. "768x768x25;512x512x49")',
    ),
    output_dir: str = typer.Option(
        ...,
        help="Output directory to save video latents",
    ),
    video_column: str = typer.Option(
        default="media_path",
        help="Column name in the dataset JSON/JSONL/CSV file containing video paths",
    ),
    batch_size: int = typer.Option(
        default=4,
        help="Batch size for processing",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    model_source: str = typer.Option(
        default=str(LtxvModelVersion.latest()),
        help="Model source - can be a version string (e.g. 'LTXV_2B_0.9.5'), HF repo, or local path",
    ),
    reshape_mode: str = typer.Option(
        default="center",
        help="How to crop videos: 'center' or 'random'",
    ),
) -> None:
    """Process videos/images and save latent representations for video generation training.

    This script processes videos and images from metadata files and saves latent representations
    that can be used for training video generation models. The output latents will maintain
    the same folder structure and naming as the corresponding media files.

    Examples:
        # Process videos from a CSV file
        python -m ltxv_trainer.process_videos dataset.csv --resolution-buckets 768x768x25 --output-dir ./latents

        # Process videos from a JSON file with custom video column
        python -m ltxv_trainer.process_videos dataset.json
            --resolution-buckets 768x768x25 --output-dir ./latents --video-column "video_path"

        # Enable VAE tiling to save GPU VRAM
        python -m ltxv_trainer.process_videos dataset.csv
            --resolution-buckets 1024x1024x25 --output-dir ./latents --vae-tiling
    """

    # Validate dataset file exists
    if not Path(dataset_file).is_file():
        raise typer.BadParameter(f"Dataset file not found: {dataset_file}")

    # Parse resolution buckets
    parsed_resolution_buckets = parse_resolution_buckets(resolution_buckets)

    if len(parsed_resolution_buckets) > 1:
        raise typer.BadParameter(
            "Multiple resolution buckets are not yet supported. Please specify only one bucket.",
            param_hint="resolution-buckets",
        )

    # Process latents
    compute_video_latents(
        dataset_file=dataset_file,
        video_column=video_column,
        resolution_buckets=parsed_resolution_buckets,
        output_dir=output_dir,
        model_source=model_source,
        reshape_mode=reshape_mode,
        batch_size=batch_size,
        device=device,
        vae_tiling=vae_tiling,
    )


if __name__ == "__main__":
    app()
