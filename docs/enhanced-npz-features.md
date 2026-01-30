# Enhanced NPZ Physics Simulation Features

This guide covers the enhanced features for working with physics simulations, including initial frame selection, channel metadata preservation, and advanced visualization options.

## 🎯 New Features

### 1. Initial Frame Selection
Specify which frame to use as the initial condition for predictions.

**Use Case**: Your simulation has 101 frames (timesteps 0-100), but you want to use timestep 10 as the initial condition and predict frames 10-110.

```yaml
data:
  npz_initial_frame_index: 10  # Use frame 10 as initial condition
```

### 2. Channel Metadata Preservation
Preserve all channel information from NPZ files for post-processing and visualization.

```yaml
data:
  npz_preserve_all_channels: true  # Save channel metadata
  npz_channel_names: ["temperature", "pressure", "velocity_x"]  # Specify channels
```

### 3. Custom Output Frame Rate
Set output video frame rate independently of simulation time.

**Use Case**: Your simulation has frames every 0.25 microseconds, but you want 20 fps output for visualization.

```yaml
validation:
  output_fps: 20.0  # Output at 20 frames per second
```

### 4. Advanced Visualization

#### Density Sum Visualization
Visualize the sum of density channels as a colorscale image.

#### Comparison GIF
Generate comparison GIFs showing:
- True simulation data
- Model prediction
- Difference between them

```yaml
validation:
  save_comparison_gif: true  # Enable comparison visualization
```

#### Custom Visualization Scripts
Use your own visualization script for specialized plots.

```yaml
validation:
  visualization_script: "/path/to/custom_viz.py"
```

## 📖 Detailed Usage

### Initial Frame Selection

When preprocessing your NPZ sequences, the trainer will now start from the specified initial frame:

```bash
# Preprocess with initial frame selection
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "512x512x101" \
    --caption-column "caption" \
    --video-column "media_path"
```

The configuration file controls which frame is used:
- `npz_initial_frame_index: 0` - Use first frame (default)
- `npz_initial_frame_index: 10` - Use 11th frame as initial condition
- Frames are loaded from `[initial_index, initial_index + num_frames_needed]`

### Working with Thousands of Simulations

For large datasets, organize your simulations:

```
simulations/
├── sim_0001/
│   ├── frame_0000.npz
│   ├── frame_0001.npz
│   └── ...
├── sim_0002/
│   └── ...
└── sim_NNNN/
    └── ...
```

Create metadata file:

```json
[
  {"caption": "Simulation 1 description", "media_path": "simulations/sim_0001"},
  {"caption": "Simulation 2 description", "media_path": "simulations/sim_0002"},
  ...
]
```

### Channel Metadata

When `npz_preserve_all_channels: true`, the preprocessor saves channel information:

```python
channel_metadata = {
    'channel_names': ['temperature', 'pressure', 'velocity_x'],
    'all_available_channels': ['temperature', 'pressure', 'velocity_x', 'velocity_y', 'density'],
    'initial_frame_index': 10,
    'num_files': 101,
    'source_path': 'simulations/sim_0001'
}
```

This metadata can be used for:
- Post-processing predicted latents
- Custom visualization
- Reconstructing multi-channel outputs

### Density Visualization

The density sum visualization combines all channels into a single colorscale image:

```python
from ltxv_trainer.npz_visualization import visualize_density_sum
import numpy as np

# Load channel data
channels = np.load('sim_0001/frame_0000.npz')
density_sum = sum(channels[ch] for ch in ['temperature', 'pressure', 'density'])

# Visualize
visualize_density_sum(
    channels=density_sum,
    output_path='density_visualization.png',
    cmap='viridis'
)
```

### Comparison GIF Generation

During validation, if `save_comparison_gif: true`, the trainer will generate animated comparisons:

```python
from ltxv_trainer.npz_visualization import create_comparison_gif

create_comparison_gif(
    true_sequence=ground_truth_data,      # (T, C, H, W)
    predicted_sequence=model_predictions,  # (T, C, H, W)
    output_path='comparison.gif',
    fps=20.0,
    visualize_as_density=True  # Sum channels for density viz
)
```

### Custom Visualization Scripts

Create a custom script for specialized visualization:

```python
#!/usr/bin/env python3
# custom_viz.py

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Read environment variables set by trainer
input_data = os.environ['INPUT_DATA']
output_path = os.environ['OUTPUT_PATH']

# Load and process data
data = np.load(input_data)

# Your custom visualization logic here
fig, ax = plt.subplots()
# ... plot your data ...
plt.savefig(output_path)
```

Specify in config:

```yaml
validation:
  visualization_script: "path/to/custom_viz.py"
```

## 🚀 Complete Workflow Example

### 1. Prepare Simulations

```bash
# Generate simulation sequences (if needed)
python scripts/create_npz_sequence.py create-example \
    --output-dir simulations \
    --num-simulations 1000 \
    --num-frames 101 \
    --resolution 512
```

### 2. Create Dataset Metadata

```python
import json
from pathlib import Path

# Create metadata for thousands of simulations
simulations = []
for sim_dir in Path('simulations').glob('simulation_*'):
    simulations.append({
        'caption': f'Physics simulation {sim_dir.name}',
        'media_path': str(sim_dir)
    })

with open('dataset.json', 'w') as f:
    json.dump(simulations, f, indent=2)

print(f"Created metadata for {len(simulations)} simulations")
```

### 3. Extract Initial Frames

```bash
# Extract frame at index 10 from each simulation for validation
python scripts/extract_initial_frames.py \
    --input-dir simulations \
    --output-dir initial_frames \
    --frame-index 10
```

### 4. Preprocess Dataset

```bash
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "512x512x101" \
    --caption-column "caption" \
    --video-column "media_path"
```

### 5. Configure Training

Edit `configs/physics_simulation_enhanced.yaml`:

```yaml
data:
  preprocessed_data_root: "simulations/.precomputed"
  npz_initial_frame_index: 10
  npz_preserve_all_channels: true
  npz_channel_names: ["temperature", "pressure", "velocity_magnitude"]

validation:
  output_fps: 20.0
  save_comparison_gif: true
  video_dims: [512, 512, 101]
  images:
    - "initial_frames/simulation_001.png"
    - "initial_frames/simulation_002.png"
    - "initial_frames/simulation_003.png"
```

### 6. Train

```bash
python scripts/train.py configs/physics_simulation_enhanced.yaml
```

## 📊 Performance Tips

### Memory Optimization

For thousands of simulations:

```yaml
data:
  num_dataloader_workers: 4  # Parallel data loading
  
optimization:
  batch_size: 1  # Start with 1 for large sequences
  gradient_accumulation_steps: 4  # Effective batch size of 4
```

### Validation Frequency

With large datasets, adjust validation frequency:

```yaml
validation:
  interval: 500  # Validate every 500 steps
  videos_per_prompt: 1  # Generate fewer samples
```

### Channel Selection

Only visualize necessary channels for speed:

```yaml
data:
  npz_channel_names: ["temperature", "pressure", "velocity_magnitude"]
  # Don't include all 10+ channels if not needed
```

## 🎨 Visualization Options

### Colormaps

Available matplotlib colormaps for density visualization:
- `viridis` - Good for general data
- `plasma` - High contrast
- `inferno` - Warm colors
- `coolwarm` - Diverging (for differences)
- `jet` - Traditional physics visualization

### GIF Settings

Customize comparison GIF generation:

```python
create_comparison_gif(
    true_sequence=true_data,
    predicted_sequence=pred_data,
    output_path='comparison.gif',
    fps=20.0,  # Playback speed
    channel_indices=[0, 1, 2],  # Which channels to visualize
    visualize_as_density=True  # Sum for density, or False for RGB
)
```

## 🔍 Debugging

### Check Channel Metadata

```python
import torch

# Load preprocessed data
data = torch.load('simulations/.precomputed/latents/latent_0000.pt')
if 'channel_metadata' in data:
    print("Channel metadata:", data['channel_metadata'])
```

### Validate Frame Selection

```bash
# Inspect which frames were loaded
python scripts/npz_utils.py validate \
    simulations/simulation_001/frame_0010.npz
```

### Test Visualization

```bash
# Test custom visualization script
export INPUT_DATA="simulations/simulation_001"
export OUTPUT_PATH="test_viz.png"
python path/to/custom_viz.py
```

## 📝 Best Practices

1. **Initial Frame Selection**: Choose a frame that represents a good starting state
2. **Channel Names**: Always specify `channels` key in NPZ files for consistency
3. **Frame Rate**: Use realistic output frame rates (10-30 fps) for smooth visualization
4. **Batch Size**: Start with 1 for long sequences, increase if memory allows
5. **Validation**: Use representative samples, not all simulations
6. **Metadata**: Enable `npz_preserve_all_channels` for post-processing flexibility

## 🆘 Troubleshooting

### "Initial frame index out of range"

```
Solution: Check that npz_initial_frame_index < number of frames in your sequences
```

### "Memory error during training"

```
Solution: 
- Reduce batch_size to 1
- Reduce video_dims
- Enable gradient_checkpointing
- Use quantization
```

### "Comparison GIF not generated"

```
Solution:
- Ensure save_comparison_gif: true
- Check that reference_videos are provided
- Verify output directory permissions
```

## 📚 API Reference

### Configuration Options

**Data Configuration**:
- `npz_initial_frame_index`: int, default 0
- `npz_channel_names`: list[str] | null
- `npz_preserve_all_channels`: bool, default False

**Validation Configuration**:
- `output_fps`: float, default 20.0
- `visualization_script`: str | null
- `save_comparison_gif`: bool, default False

### Python API

```python
from ltxv_trainer.npz_visualization import (
    visualize_density_sum,
    create_comparison_gif,
    run_custom_visualization_script,
    load_npz_sequence,
    save_channels_as_npz
)
```

See module docstrings for detailed API documentation.
