#!/usr/bin/env python3
"""
Extract initial frames from NPZ sequences for validation.

This script extracts a specific frame (e.g., frame 10) from each NPZ sequence
and saves it as an image for use in image-to-video validation.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import typer
from rich.console import Console
from rich.progress import track

app = typer.Typer(help="Extract initial frames from NPZ sequences")
console = Console()


def extract_frame_from_npz_sequence(
    sequence_dir: Path,
    frame_index: int,
    output_path: Path,
    channel_names: list[str] | None = None,
    visualize_as_density: bool = True
) -> None:
    """Extract a specific frame from an NPZ sequence directory.
    
    Args:
        sequence_dir: Directory containing NPZ files
        frame_index: Which frame to extract (0-indexed)
        output_path: Where to save the extracted frame image
        channel_names: Which channels to use (None = use first 3 or 'channels' key)
        visualize_as_density: If True, sum all channels; if False, use first 3 as RGB
    """
    # Get all NPZ files
    npz_files = sorted(list(sequence_dir.glob('*.npz')))
    
    if not npz_files:
        console.print(f"[red]No NPZ files found in {sequence_dir}[/red]")
        return
    
    if frame_index >= len(npz_files):
        console.print(f"[red]Frame index {frame_index} >= number of files {len(npz_files)}[/red]")
        return
    
    # Load the specified frame
    frame_file = npz_files[frame_index]
    frame_data = np.load(frame_file)
    
    # Get channel names
    available_keys = [k for k in frame_data.keys() if k not in ['fps', 'channels', 'frames', 'data']]
    
    if channel_names is None:
        if 'channels' in frame_data:
            channel_names = frame_data['channels']
            if isinstance(channel_names, np.ndarray):
                channel_names = channel_names.tolist()
                if isinstance(channel_names[0], bytes):
                    channel_names = [c.decode('utf-8') for c in channel_names]
        else:
            channel_names = available_keys[:3]
    
    # Load channel data
    channels = []
    for ch_name in channel_names:
        if ch_name in frame_data:
            channels.append(frame_data[ch_name])
        else:
            console.print(f"[yellow]Warning: Channel '{ch_name}' not found in {frame_file.name}[/yellow]")
    
    if not channels:
        console.print(f"[red]No valid channels found in {frame_file.name}[/red]")
        return
    
    # Stack channels
    channel_array = np.stack(channels, axis=0)  # (C, H, W)
    
    # Create visualization
    if visualize_as_density:
        # Sum all channels for density
        image_data = channel_array.sum(axis=0)  # (H, W)
        
        # Normalize to [0, 255]
        if image_data.max() > image_data.min():
            image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
        image_data = (image_data * 255).astype(np.uint8)
        
        # Convert to RGB by repeating
        image_data = np.stack([image_data] * 3, axis=-1)
    else:
        # Use first 3 channels as RGB
        if channel_array.shape[0] == 1:
            # Grayscale - repeat to RGB
            image_data = np.repeat(channel_array[0:1], 3, axis=0).transpose(1, 2, 0)
        elif channel_array.shape[0] == 2:
            # 2 channels - add zero channel
            zero_ch = np.zeros_like(channel_array[0:1])
            image_data = np.concatenate([channel_array, zero_ch], axis=0).transpose(1, 2, 0)
        else:
            # 3+ channels - take first 3
            image_data = channel_array[:3].transpose(1, 2, 0)  # (H, W, 3)
        
        # Normalize to [0, 255]
        if image_data.max() > 1.0:
            if image_data.max() <= 255.0:
                image_data = image_data.astype(np.uint8)
            else:
                image_data = (image_data / image_data.max() * 255).astype(np.uint8)
        else:
            image_data = (image_data * 255).astype(np.uint8)
    
    # Save as PNG
    img = Image.fromarray(image_data, mode='RGB')
    img.save(output_path)


@app.command()
def extract_batch(
    input_dir: Path = typer.Argument(..., help="Directory containing simulation subdirectories"),
    output_dir: Path = typer.Argument(..., help="Directory to save extracted frames"),
    frame_index: int = typer.Option(0, help="Frame index to extract (0-indexed)"),
    visualize_as_density: bool = typer.Option(True, help="Sum channels for density visualization"),
    pattern: str = typer.Option("*", help="Pattern to match simulation directories"),
) -> None:
    """Extract initial frames from multiple NPZ sequences."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all simulation directories
    sim_dirs = [d for d in input_dir.glob(pattern) if d.is_dir()]
    
    if not sim_dirs:
        console.print(f"[red]No simulation directories found in {input_dir}[/red]")
        return
    
    console.print(f"\n[green]Found {len(sim_dirs)} simulation directories[/green]")
    console.print(f"Extracting frame {frame_index} from each sequence...\n")
    
    success_count = 0
    for sim_dir in track(sim_dirs, description="Extracting frames"):
        output_path = output_dir / f"{sim_dir.name}.png"
        
        try:
            extract_frame_from_npz_sequence(
                sequence_dir=sim_dir,
                frame_index=frame_index,
                output_path=output_path,
                visualize_as_density=visualize_as_density
            )
            success_count += 1
        except Exception as e:
            console.print(f"[red]Error processing {sim_dir.name}: {e}[/red]")
    
    console.print(f"\n[green]✓ Successfully extracted {success_count}/{len(sim_dirs)} frames[/green]")
    console.print(f"Saved to: {output_dir}\n")


@app.command()
def extract_single(
    sequence_dir: Path = typer.Argument(..., help="NPZ sequence directory"),
    output_path: Path = typer.Argument(..., help="Output image path"),
    frame_index: int = typer.Option(0, help="Frame index to extract (0-indexed)"),
    channels: list[str] = typer.Option(None, help="Specific channels to use"),
    visualize_as_density: bool = typer.Option(True, help="Sum channels for density visualization"),
) -> None:
    """Extract a single frame from an NPZ sequence."""
    
    console.print(f"\n[cyan]Extracting frame {frame_index} from {sequence_dir}[/cyan]")
    
    try:
        extract_frame_from_npz_sequence(
            sequence_dir=sequence_dir,
            frame_index=frame_index,
            output_path=output_path,
            channel_names=channels if channels else None,
            visualize_as_density=visualize_as_density
        )
        console.print(f"[green]✓ Saved to {output_path}[/green]\n")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]\n")
        raise typer.Exit(1)


@app.command()
def info(
    sequence_dir: Path = typer.Argument(..., help="NPZ sequence directory"),
) -> None:
    """Display information about an NPZ sequence."""
    
    npz_files = sorted(list(sequence_dir.glob('*.npz')))
    
    if not npz_files:
        console.print(f"[red]No NPZ files found in {sequence_dir}[/red]")
        return
    
    console.print(f"\n[bold cyan]NPZ Sequence Information[/bold cyan]")
    console.print(f"Directory: {sequence_dir}")
    console.print(f"Number of frames: {len(npz_files)}")
    
    # Load first file for details
    first_data = np.load(npz_files[0])
    
    console.print(f"\n[bold]First frame ({npz_files[0].name}):[/bold]")
    console.print(f"Keys: {list(first_data.keys())}")
    
    # Get channels
    available_keys = [k for k in first_data.keys() if k not in ['fps', 'channels', 'frames', 'data']]
    
    if 'channels' in first_data:
        channel_names = first_data['channels']
        if isinstance(channel_names, np.ndarray):
            channel_names = channel_names.tolist()
        console.print(f"Specified channels: {channel_names}")
    
    console.print(f"Available channel keys: {available_keys}")
    
    # Display first channel shape
    if available_keys:
        first_channel = first_data[available_keys[0]]
        console.print(f"\nChannel shape: {first_channel.shape}")
        console.print(f"Data type: {first_channel.dtype}")
        console.print(f"Value range: [{first_channel.min():.4f}, {first_channel.max():.4f}]")
    
    if 'fps' in first_data:
        console.print(f"\nFPS: {first_data['fps']}")
    
    # Check consistency across frames
    if len(npz_files) > 1:
        console.print(f"\n[bold]Checking consistency...[/bold]")
        last_data = np.load(npz_files[-1])
        if list(first_data.keys()) != list(last_data.keys()):
            console.print("[yellow]⚠ Warning: Keys differ between first and last frame[/yellow]")
        else:
            console.print("[green]✓ Keys consistent across frames[/green]")
    
    console.print()


if __name__ == "__main__":
    app()
