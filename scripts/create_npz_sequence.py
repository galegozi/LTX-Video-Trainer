#!/usr/bin/env python3
"""
Example script: Create NPZ sequence from physics simulation data.

This script demonstrates how to convert physics simulation output
into the NPZ sequence format expected by the LTX-Video trainer.
"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import typer
from rich.console import Console
from rich.progress import track

app = typer.Typer(help="Create NPZ sequences from physics simulation data")
console = Console()


def create_example_simulation_data(
    num_frames: int = 121,
    height: int = 512,
    width: int = 512
) -> Dict[str, np.ndarray]:
    """Generate example physics simulation data.
    
    In practice, this would be replaced with actual simulation output.
    
    Returns:
        Dictionary with channel names as keys and (T, H, W) arrays as values
    """
    console.print(f"[cyan]Generating example simulation data ({num_frames} frames, {height}x{width})...[/cyan]")
    
    # Simulate temperature evolution
    temperature = np.random.rand(num_frames, height, width).astype(np.float32)
    # Add some structure - temperature diffusion
    for t in range(1, num_frames):
        temperature[t] = 0.9 * temperature[t] + 0.1 * temperature[t-1]
    
    # Simulate pressure field
    pressure = np.random.rand(num_frames, height, width).astype(np.float32)
    
    # Simulate velocity field
    velocity_x = (np.random.rand(num_frames, height, width) - 0.5).astype(np.float32)
    velocity_y = (np.random.rand(num_frames, height, width) - 0.5).astype(np.float32)
    
    # Compute velocity magnitude
    velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2).astype(np.float32)
    
    # Simulate vorticity
    vorticity = np.gradient(velocity_x, axis=2) - np.gradient(velocity_y, axis=1)
    vorticity = vorticity.astype(np.float32)
    
    return {
        'temperature': temperature,
        'pressure': pressure,
        'velocity_x': velocity_x,
        'velocity_y': velocity_y,
        'velocity_magnitude': velocity_magnitude,
        'vorticity': vorticity,
    }


def create_npz_sequence(
    simulation_data: Dict[str, np.ndarray],
    output_dir: Path,
    channel_names: List[str],
    fps: float = 24.0,
    normalize_per_channel: bool = True
) -> None:
    """Convert simulation data to NPZ sequence format.
    
    Args:
        simulation_data: Dictionary with channel names as keys, (T, H, W) arrays as values
        output_dir: Directory to save NPZ files
        channel_names: List of channel names to include (in RGB order)
        fps: Frames per second
        normalize_per_channel: If True, normalize each channel independently to [0, 1]
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get number of frames
    num_frames = next(iter(simulation_data.values())).shape[0]
    
    console.print(f"\n[green]Creating NPZ sequence in {output_dir}[/green]")
    console.print(f"  Frames: {num_frames}")
    console.print(f"  Channels: {channel_names}")
    console.print(f"  FPS: {fps}")
    console.print(f"  Normalize per channel: {normalize_per_channel}\n")
    
    # Normalize channels if requested
    if normalize_per_channel:
        console.print("[cyan]Normalizing channels to [0, 1] range...[/cyan]")
        for channel_name in channel_names:
            if channel_name in simulation_data:
                data = simulation_data[channel_name]
                min_val = data.min()
                max_val = data.max()
                if max_val > min_val:
                    simulation_data[channel_name] = (data - min_val) / (max_val - min_val)
                console.print(f"  ✓ {channel_name}: [{min_val:.3f}, {max_val:.3f}] → [0.0, 1.0]")
    
    # Create NPZ files
    console.print("\n[cyan]Creating NPZ files...[/cyan]")
    for frame_idx in track(range(num_frames), description="Saving frames"):
        frame_data = {}
        
        # Extract each channel for this frame
        for channel_name in channel_names:
            if channel_name not in simulation_data:
                raise ValueError(f"Channel '{channel_name}' not found in simulation_data")
            frame_data[channel_name] = simulation_data[channel_name][frame_idx]
        
        # Add metadata
        frame_data['channels'] = np.array(channel_names, dtype='U32')
        frame_data['fps'] = fps
        
        # Save frame
        output_path = output_dir / f'frame_{frame_idx:04d}.npz'
        np.savez(output_path, **frame_data)
    
    console.print(f"\n[green]✓ Successfully created {num_frames} NPZ files[/green]")


def create_dataset_metadata(
    simulation_dirs: List[Path],
    captions: List[str],
    output_file: Path
) -> None:
    """Create dataset JSON metadata file.
    
    Args:
        simulation_dirs: List of paths to directories containing NPZ sequences
        captions: List of captions for each simulation
        output_file: Output JSON file path
    """
    console.print(f"\n[cyan]Creating dataset metadata...[/cyan]")
    
    dataset = []
    
    for sim_dir, caption in zip(simulation_dirs, captions):
        # Verify the directory has NPZ files
        npz_files = list(Path(sim_dir).glob('*.npz'))
        if npz_files:
            dataset.append({
                'caption': caption,
                'media_path': str(sim_dir)
            })
            console.print(f"  ✓ Added {sim_dir} with {len(npz_files)} frames")
        else:
            console.print(f"  [yellow]⚠ Skipped {sim_dir} (no NPZ files found)[/yellow]")
    
    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    console.print(f"\n[green]✓ Created dataset metadata: {output_file}[/green]")
    console.print(f"  Total simulations: {len(dataset)}")


@app.command()
def create_example(
    output_dir: Path = typer.Option("example_simulations", help="Output directory for example data"),
    num_simulations: int = typer.Option(3, help="Number of example simulations to create"),
    num_frames: int = typer.Option(25, help="Number of frames per simulation"),
    resolution: int = typer.Option(256, help="Resolution (height and width)"),
) -> None:
    """Create example NPZ sequences for testing."""
    
    console.print("\n[bold cyan]Creating Example NPZ Sequences[/bold cyan]\n")
    
    # Create multiple simulation sequences
    sim_dirs = []
    captions = [
        "High viscosity fluid dynamics with turbulent flow and vortex formation",
        "Low viscosity fluid dynamics with laminar flow patterns",
        "Turbulent flow simulation showing temperature and pressure gradients",
    ]
    
    for i in range(num_simulations):
        # Generate simulation data
        sim_data = create_example_simulation_data(
            num_frames=num_frames,
            height=resolution,
            width=resolution
        )
        
        # Create NPZ sequence
        sim_dir = output_dir / f"simulation_{i+1:03d}"
        create_npz_sequence(
            simulation_data=sim_data,
            output_dir=sim_dir,
            channel_names=['temperature', 'pressure', 'velocity_magnitude'],
            fps=24.0,
            normalize_per_channel=True
        )
        
        sim_dirs.append(sim_dir)
    
    # Create dataset metadata
    dataset_file = output_dir / "dataset.json"
    create_dataset_metadata(
        simulation_dirs=sim_dirs,
        captions=captions[:num_simulations],
        output_file=dataset_file
    )
    
    console.print("\n[bold green]✓ Example data created successfully![/bold green]")
    console.print(f"\n[cyan]Next steps:[/cyan]")
    console.print(f"  1. Inspect the data: ls {output_dir}/")
    console.print(f"  2. Preprocess: python scripts/preprocess_dataset.py {dataset_file} --resolution-buckets \"{resolution}x{resolution}x{num_frames}\"")
    console.print(f"  3. Train: python scripts/train.py configs/physics_simulation_npz.yaml")
    console.print()


@app.command()
def convert(
    input_dir: Path = typer.Argument(..., help="Directory containing simulation output files"),
    output_dir: Path = typer.Argument(..., help="Output directory for NPZ sequence"),
    channel_names: List[str] = typer.Option(
        ["channel1", "channel2", "channel3"],
        "--channel",
        "-c",
        help="Channel names to include (in order for RGB mapping)"
    ),
    fps: float = typer.Option(24.0, help="Frames per second"),
    file_pattern: str = typer.Option("*.npy", help="File pattern for input files"),
) -> None:
    """Convert simulation output files to NPZ sequence format.
    
    This is a template - modify based on your simulation output format.
    """
    console.print("\n[bold yellow]⚠ This is a template function[/bold yellow]")
    console.print("Modify this function based on your simulation output format.\n")
    
    # Example: Load .npy files
    input_files = sorted(list(input_dir.glob(file_pattern)))
    
    if not input_files:
        console.print(f"[red]No files found matching pattern '{file_pattern}' in {input_dir}[/red]")
        raise typer.Exit(1)
    
    console.print(f"Found {len(input_files)} input files")
    console.print("\n[cyan]Example conversion code:[/cyan]")
    console.print("""
    # Load your simulation data
    simulation_data = {}
    for channel in channel_names:
        # Load channel data from your files
        # This depends on your file format
        channel_data = np.load(f'{input_dir}/{channel}.npy')
        simulation_data[channel] = channel_data
    
    # Create NPZ sequence
    create_npz_sequence(
        simulation_data=simulation_data,
        output_dir=output_dir,
        channel_names=channel_names,
        fps=fps
    )
    """)


if __name__ == "__main__":
    app()
