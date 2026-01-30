"""
NPZ-specific visualization utilities for physics simulations.

This module provides custom visualization for multi-channel NPZ data,
including density visualization, comparison views, and custom scripts.
"""

import subprocess
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio

from ltxv_trainer import logger


def visualize_density_sum(
    channels: np.ndarray,
    output_path: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis'
) -> None:
    """Visualize sum of density channels as colorscale image.
    
    Args:
        channels: Array of shape (C, H, W) or (H, W) 
        output_path: Path to save visualization
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cmap: Matplotlib colormap name
    """
    if channels.ndim == 3:
        # Sum across all channels
        density_sum = channels.sum(axis=0)
    else:
        density_sum = channels
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(density_sum, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_comparison_gif(
    true_sequence: np.ndarray,
    predicted_sequence: np.ndarray,
    output_path: Path,
    fps: float = 20.0,
    channel_indices: Optional[List[int]] = None,
    visualize_as_density: bool = True
) -> None:
    """Create comparison GIF showing true, predicted, and difference.
    
    Args:
        true_sequence: Ground truth with shape (T, C, H, W) or (T, H, W)
        predicted_sequence: Predicted with shape (T, C, H, W) or (T, H, W)
        output_path: Path to save GIF
        fps: Frames per second for output
        channel_indices: Which channels to visualize (if None, sum all)
        visualize_as_density: If True, sum channels for density visualization
    """
    num_frames = min(len(true_sequence), len(predicted_sequence))
    
    frames = []
    for t in range(num_frames):
        true_frame = true_sequence[t]
        pred_frame = predicted_sequence[t]
        
        # Process frames
        if true_frame.ndim == 3:  # (C, H, W)
            if visualize_as_density:
                # Sum channels for density
                if channel_indices is not None:
                    true_vis = true_frame[channel_indices].sum(axis=0)
                    pred_vis = pred_frame[channel_indices].sum(axis=0)
                else:
                    true_vis = true_frame.sum(axis=0)
                    pred_vis = pred_frame.sum(axis=0)
            else:
                # Convert first 3 channels to RGB
                true_vis = true_frame[:3].transpose(1, 2, 0)
                pred_vis = pred_frame[:3].transpose(1, 2, 0)
        else:  # (H, W)
            true_vis = true_frame
            pred_vis = pred_frame
        
        # Normalize for consistent scaling across frames
        vmin = min(true_vis.min(), pred_vis.min())
        vmax = max(true_vis.max(), pred_vis.max())
        
        # Compute difference
        diff_vis = pred_vis - true_vis
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # True
        im0 = axes[0].imshow(true_vis, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'True (Frame {t})')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Predicted
        im1 = axes[1].imshow(pred_vis, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Predicted (Frame {t})')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Difference
        diff_max = max(abs(diff_vis.min()), abs(diff_vis.max()))
        im2 = axes[2].imshow(diff_vis, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        axes[2].set_title(f'Difference (Frame {t})')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Convert to image array
        fig.canvas.draw()
        frame_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame_data = frame_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame_data)
        
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    logger.info(f"Saved comparison GIF with {len(frames)} frames to {output_path}")


def run_custom_visualization_script(
    script_path: Path,
    input_data_path: Path,
    output_path: Path,
    **kwargs
) -> bool:
    """Run custom visualization script.
    
    Args:
        script_path: Path to custom visualization script
        input_data_path: Path to input data (NPZ file or directory)
        output_path: Path for output visualization
        **kwargs: Additional arguments passed as environment variables
    
    Returns:
        True if script executed successfully
    """
    if not script_path.exists():
        logger.error(f"Custom visualization script not found: {script_path}")
        return False
    
    # Prepare environment variables
    env = {
        'INPUT_DATA': str(input_data_path),
        'OUTPUT_PATH': str(output_path),
    }
    
    # Add any additional kwargs as env vars
    for key, value in kwargs.items():
        env[key.upper()] = str(value)
    
    try:
        # Run script
        result = subprocess.run(
            ['python', str(script_path)],
            env={**subprocess.os.environ, **env},
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"Custom visualization script completed successfully")
            return True
        else:
            logger.error(f"Custom visualization script failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Custom visualization script timed out after 300 seconds")
        return False
    except Exception as e:
        logger.error(f"Error running custom visualization script: {e}")
        return False


def extract_channel_data(
    npz_path: Path,
    channel_names: Optional[List[str]] = None
) -> tuple[np.ndarray, List[str]]:
    """Extract channel data from NPZ file.
    
    Args:
        npz_path: Path to NPZ file
        channel_names: Specific channels to extract (None = all)
    
    Returns:
        Tuple of (channel_data, channel_names) where channel_data has shape (C, H, W)
    """
    data = np.load(npz_path)
    
    # Get available channels
    available_keys = [k for k in data.keys() if k not in ['fps', 'channels', 'frames', 'data']]
    
    # Determine which channels to use
    if channel_names is None:
        if 'channels' in data:
            channel_names = data['channels'].tolist()
            if isinstance(channel_names[0], bytes):
                channel_names = [c.decode('utf-8') for c in channel_names]
        else:
            channel_names = available_keys
    
    # Extract channel data
    channels = []
    for ch_name in channel_names:
        if ch_name in data:
            channels.append(data[ch_name])
        else:
            logger.warning(f"Channel '{ch_name}' not found in {npz_path}")
    
    if not channels:
        raise ValueError(f"No valid channels found in {npz_path}")
    
    channel_data = np.stack(channels, axis=0)  # (C, H, W)
    return channel_data, channel_names


def load_npz_sequence(
    sequence_dir: Path,
    start_frame: int = 0,
    num_frames: Optional[int] = None,
    channel_names: Optional[List[str]] = None
) -> tuple[np.ndarray, List[str], float]:
    """Load NPZ sequence from directory.
    
    Args:
        sequence_dir: Directory containing NPZ files
        start_frame: Starting frame index
        num_frames: Number of frames to load (None = all)
        channel_names: Specific channels to extract
    
    Returns:
        Tuple of (sequence_data, channel_names, fps)
        where sequence_data has shape (T, C, H, W)
    """
    npz_files = sorted(list(sequence_dir.glob('*.npz')))
    
    if not npz_files:
        raise ValueError(f"No NPZ files found in {sequence_dir}")
    
    # Load first file to get fps and channel info
    first_data = np.load(npz_files[start_frame])
    fps = float(first_data.get('fps', 24.0))
    
    # Determine frames to load
    if num_frames is None:
        num_frames = len(npz_files) - start_frame
    else:
        num_frames = min(num_frames, len(npz_files) - start_frame)
    
    # Load frames
    frames = []
    extracted_channel_names = None
    
    for i in range(start_frame, start_frame + num_frames):
        if i >= len(npz_files):
            break
        
        channel_data, ch_names = extract_channel_data(npz_files[i], channel_names)
        frames.append(channel_data)
        
        if extracted_channel_names is None:
            extracted_channel_names = ch_names
    
    sequence_data = np.stack(frames, axis=0)  # (T, C, H, W)
    return sequence_data, extracted_channel_names, fps


def save_channels_as_npz(
    channel_data: np.ndarray,
    channel_names: List[str],
    output_path: Path,
    fps: float = 24.0
) -> None:
    """Save channel data to NPZ file.
    
    Args:
        channel_data: Array of shape (C, H, W) or (T, C, H, W)
        channel_names: Names of channels
        output_path: Output NPZ file path
        fps: Frames per second
    """
    save_dict = {}
    
    if channel_data.ndim == 3:
        # Single frame (C, H, W)
        for i, name in enumerate(channel_names):
            save_dict[name] = channel_data[i]
    elif channel_data.ndim == 4:
        # Sequence (T, C, H, W)
        for i, name in enumerate(channel_names):
            save_dict[name] = channel_data[:, i, :, :]
    
    save_dict['channels'] = np.array(channel_names, dtype='U32')
    save_dict['fps'] = fps
    
    np.savez(output_path, **save_dict)
    logger.info(f"Saved {len(channel_names)} channels to {output_path}")
