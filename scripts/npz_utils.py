#!/usr/bin/env python3
"""
Utility script for working with NPZ physics simulation files.

This script provides helper functions for:
- Extracting first frames from NPZ files for image-to-video validation
- Validating NPZ file format
- Converting various formats to the expected NPZ structure
"""

from pathlib import Path
import numpy as np
from PIL import Image
import typer
from rich.console import Console
from rich.progress import track

app = typer.Typer(help="Utilities for working with NPZ physics simulation files")
console = Console()


def extract_first_frame(npz_path: Path, output_path: Path) -> None:
    """Extract and save the first frame from an NPZ file.
    
    Args:
        npz_path: Path to the input NPZ file
        output_path: Path where the first frame image will be saved
    """
    data = np.load(npz_path)
    
    # Get frames - support both 'frames' and 'data' keys
    frames = data['frames'] if 'frames' in data else data['data']
    
    # Get first frame
    first_frame = frames[0]
    
    # Handle different formats
    if first_frame.ndim == 3 and first_frame.shape[-1] in [1, 3, 4]:
        # (H, W, C) format - already correct
        pass
    elif first_frame.ndim == 3 and first_frame.shape[0] in [1, 3, 4]:
        # (C, H, W) format - convert to (H, W, C)
        first_frame = np.transpose(first_frame, (1, 2, 0))
    elif first_frame.ndim == 2:
        # (H, W) grayscale - expand to RGB
        first_frame = np.stack([first_frame] * 3, axis=-1)
    else:
        raise ValueError(f"Unexpected frame shape: {first_frame.shape}")
    
    # Normalize to [0, 255] if needed
    if first_frame.max() <= 1.0:
        first_frame = (first_frame * 255).astype(np.uint8)
    else:
        first_frame = first_frame.astype(np.uint8)
    
    # Handle alpha channel
    if first_frame.shape[-1] == 4:
        # Drop alpha channel for consistency
        first_frame = first_frame[:, :, :3]
    
    # Ensure we have 3 channels
    if first_frame.shape[-1] == 1:
        first_frame = np.repeat(first_frame, 3, axis=-1)
    
    # Save as PNG
    img = Image.fromarray(first_frame, mode='RGB')
    img.save(output_path)


@app.command()
def extract_frames(
    input_dir: Path = typer.Argument(..., help="Directory containing NPZ files"),
    output_dir: Path = typer.Argument(..., help="Directory to save first frame images"),
    pattern: str = typer.Option("*.npz", help="Pattern to match NPZ files"),
) -> None:
    """Extract first frames from all NPZ files in a directory."""
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all NPZ files
    npz_files = list(input_dir.glob(pattern))
    
    if not npz_files:
        console.print(f"[yellow]No NPZ files found in {input_dir} matching pattern '{pattern}'[/yellow]")
        return
    
    console.print(f"[green]Found {len(npz_files)} NPZ files[/green]")
    
    # Extract first frame from each file
    for npz_path in track(npz_files, description="Extracting first frames..."):
        output_path = output_dir / f"{npz_path.stem}.png"
        try:
            extract_first_frame(npz_path, output_path)
        except Exception as e:
            console.print(f"[red]Error processing {npz_path.name}: {e}[/red]")
    
    console.print(f"[green]✓ First frames saved to {output_dir}[/green]")


@app.command()
def validate(
    npz_path: Path = typer.Argument(..., help="Path to NPZ file to validate"),
) -> None:
    """Validate an NPZ file format for compatibility with LTX-Video trainer."""
    
    console.print(f"\n[bold]Validating {npz_path.name}[/bold]\n")
    
    try:
        data = np.load(npz_path)
    except Exception as e:
        console.print(f"[red]✗ Failed to load NPZ file: {e}[/red]")
        raise typer.Exit(1)
    
    # Check for required keys
    console.print("[bold]Keys in NPZ file:[/bold]")
    for key in data.keys():
        console.print(f"  - {key}")
    
    # Check for frames data
    frames_key = None
    if 'frames' in data:
        frames_key = 'frames'
        console.print("\n[green]✓ Found 'frames' key[/green]")
    elif 'data' in data:
        frames_key = 'data'
        console.print("\n[green]✓ Found 'data' key[/green]")
    else:
        console.print("\n[red]✗ Missing required 'frames' or 'data' key[/red]")
        raise typer.Exit(1)
    
    # Validate frames
    frames = data[frames_key]
    console.print(f"\n[bold]Frame data:[/bold]")
    console.print(f"  Shape: {frames.shape}")
    console.print(f"  Dtype: {frames.dtype}")
    console.print(f"  Value range: [{frames.min():.4f}, {frames.max():.4f}]")
    
    # Check dimensions
    if frames.ndim not in [3, 4]:
        console.print(f"[red]✗ Invalid dimensions: expected 3 or 4, got {frames.ndim}[/red]")
        raise typer.Exit(1)
    
    if frames.ndim == 4:
        # Check if format is (T, H, W, C) or (T, C, H, W)
        if frames.shape[-1] in [1, 3, 4]:
            console.print(f"[green]✓ Format detected as (T, H, W, C)[/green]")
            num_frames, height, width, channels = frames.shape
        elif frames.shape[1] in [1, 3, 4]:
            console.print(f"[green]✓ Format detected as (T, C, H, W)[/green]")
            num_frames, channels, height, width = frames.shape
        else:
            console.print(f"[yellow]⚠ Unusual shape - might cause issues[/yellow]")
            num_frames = frames.shape[0]
            height = frames.shape[2] if frames.shape[1] in [1, 3, 4] else frames.shape[1]
            width = frames.shape[3] if frames.shape[1] in [1, 3, 4] else frames.shape[2]
            channels = frames.shape[1] if frames.shape[1] in [1, 3, 4] else "?"
    else:
        # Grayscale (T, H, W)
        console.print(f"[green]✓ Format detected as (T, H, W) - grayscale[/green]")
        num_frames, height, width = frames.shape
        channels = 1
    
    console.print(f"\n[bold]Video properties:[/bold]")
    console.print(f"  Frames: {num_frames}")
    console.print(f"  Resolution: {width}x{height}")
    console.print(f"  Channels: {channels}")
    
    # Check FPS
    if 'fps' in data:
        fps = data['fps']
        console.print(f"  FPS: {fps}")
    else:
        console.print(f"  FPS: Not specified (will default to 24)")
    
    # Validate frame count
    valid_frame_counts = [8*n + 1 for n in range(1, 25)]  # 9, 17, 25, ..., 193
    if num_frames not in valid_frame_counts:
        console.print(f"\n[yellow]⚠ Frame count {num_frames} is not optimal[/yellow]")
        console.print(f"  Recommended frame counts: {valid_frame_counts[:10]}...")
        console.print(f"  The preprocessor will truncate to the nearest valid count")
    else:
        console.print(f"\n[green]✓ Frame count {num_frames} is valid[/green]")
    
    # Validate value range
    if frames.max() > 1.0 and frames.max() <= 255.0:
        console.print(f"[green]✓ Values in [0, 255] range - will be normalized[/green]")
    elif frames.max() <= 1.0:
        console.print(f"[green]✓ Values already normalized to [0, 1][/green]")
    else:
        console.print(f"[yellow]⚠ Unusual value range - might cause issues[/yellow]")
    
    console.print(f"\n[bold green]✓ Validation complete - NPZ file is compatible[/bold green]\n")


@app.command()
def create_example(
    output_path: Path = typer.Argument(..., help="Path where example NPZ will be saved"),
    frames: int = typer.Option(25, help="Number of frames"),
    width: int = typer.Option(512, help="Frame width"),
    height: int = typer.Option(512, help="Frame height"),
    fps: float = typer.Option(24.0, help="Frames per second"),
) -> None:
    """Create an example NPZ file with random data for testing."""
    
    console.print(f"Creating example NPZ file with {frames} frames at {width}x{height}...")
    
    # Generate random simulation data
    # Using (T, H, W, C) format
    simulation_frames = np.random.rand(frames, height, width, 3).astype(np.float32)
    
    # Save as NPZ
    np.savez(output_path, frames=simulation_frames, fps=fps)
    
    console.print(f"[green]✓ Example NPZ file saved to {output_path}[/green]")
    console.print(f"  Frames: {frames}")
    console.print(f"  Resolution: {width}x{height}")
    console.print(f"  FPS: {fps}")
    console.print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


@app.command()
def convert(
    input_path: Path = typer.Argument(..., help="Path to input file (video or image sequence)"),
    output_path: Path = typer.Argument(..., help="Path where NPZ will be saved"),
    fps: float = typer.Option(24.0, help="Frames per second"),
    max_frames: int = typer.Option(None, help="Maximum number of frames to convert"),
) -> None:
    """Convert a video file or image sequence to NPZ format.
    
    Note: This is a basic converter. For production use, consider using ffmpeg
    or other specialized tools to ensure proper frame extraction.
    """
    try:
        import cv2
    except ImportError:
        console.print("[red]Error: opencv-python is required for video conversion[/red]")
        console.print("Install it with: pip install opencv-python")
        raise typer.Exit(1)
    
    console.print(f"Converting {input_path} to NPZ format...")
    
    # Read video
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        console.print(f"[red]Failed to open video file: {input_path}[/red]")
        raise typer.Exit(1)
    
    # Get video properties
    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    console.print(f"  Total frames: {total_frames}")
    console.print(f"  Source FPS: {source_fps:.2f}")
    
    # Read all frames
    frames = []
    for _ in track(range(total_frames), description="Reading frames..."):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    
    cap.release()
    
    # Stack frames into array (T, H, W, C)
    frames_array = np.stack(frames, axis=0)
    
    # Save as NPZ
    np.savez(output_path, frames=frames_array, fps=fps)
    
    console.print(f"[green]✓ Converted {len(frames)} frames to {output_path}[/green]")
    console.print(f"  Shape: {frames_array.shape}")
    console.print(f"  FPS: {fps}")


if __name__ == "__main__":
    app()
