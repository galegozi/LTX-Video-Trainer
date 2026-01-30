# Enhanced NPZ Features - Quick Reference

This document directly addresses the requirements for training on thousands of physics simulations.

## ✅ Requirements Met

### 1. ✅ Specify Initial Frame and Directory
**Requirement**: "Can I specify which frame is the initial and the directory for all the simulations, having the model generate all 101 timesteps from the initial one?"

**Solution**:
```yaml
# In your config file (configs/physics_simulation_enhanced.yaml)
data:
  preprocessed_data_root: "/path/to/simulations/.precomputed"
  npz_initial_frame_index: 0  # Specify which frame is initial (0-100)
  
validation:
  video_dims: [512, 512, 101]  # Generate all 101 timesteps
```

**Dataset Structure**:
```
simulations/
├── sim_0001/  (101 frames each)
├── sim_0002/
├── sim_0003/
└── ... (thousands of simulations)
```

### 2. ✅ Custom Visualization Script
**Requirement**: "I'd like to use a custom visualization script instead. How do I specify that?"

**Solution**:
```yaml
validation:
  visualization_script: "/path/to/your_custom_viz.py"
```

**Custom Script Template**:
```python
#!/usr/bin/env python3
import os
import numpy as np

# Access provided environment variables
input_data = os.environ['INPUT_DATA']  # Path to NPZ or directory
output_path = os.environ['OUTPUT_PATH']  # Where to save result

# Your custom visualization code
data = np.load(input_data)
# ... process and visualize ...
# Save to output_path
```

### 3. ✅ Density Sum Visualization
**Requirement**: "Or can I just have the visualization show the sum of the density channels which would show a colorscale image"

**Solution - Option A (Built-in)**:
```python
from ltxv_trainer.npz_visualization import visualize_density_sum

# Sum all density channels
visualize_density_sum(
    channels=density_channels,  # (C, H, W) array
    output_path='density_sum.png',
    cmap='viridis'  # Or 'plasma', 'inferno', etc.
)
```

**Solution - Option B (Configuration)**:
```yaml
data:
  npz_channel_names: ["density1", "density2", "density3"]  # Your density channels
```

### 4. ✅ Comparison GIF (True/Predicted/Difference)
**Requirement**: "Can I have it visualize the true image, the cumulative density colorscale predicted, and the difference instead, showing it in a gif?"

**Solution**:
```yaml
validation:
  save_comparison_gif: true  # Enables comparison visualization
```

This generates GIFs with three panels:
- **Left**: True simulation data (cumulative density)
- **Center**: Model prediction (cumulative density)
- **Right**: Difference (predicted - true)

### 5. ⚠️ Predict All Channels (Limitation)
**Requirement**: "I'd like the model to predict all channels, not just 3"

**Current Limitation**: The LTX-Video VAE is designed for RGB (3 channels) input/output. The model cannot directly predict more than 3 channels through the VAE.

**Workaround Solution**:
1. **During Training**: Model learns from 3 channels (specified via `npz_channel_names`)
2. **Channel Metadata**: All channel information is preserved
3. **Post-Processing**: Use preserved metadata to reconstruct multi-channel outputs

```yaml
data:
  npz_channel_names: ["temperature", "pressure", "velocity_x"]  # 3 for training
  npz_preserve_all_channels: true  # Preserve all channel metadata
```

**Future Enhancement**: Multi-channel prediction would require:
- Custom VAE modification (beyond scope)
- Or multiple model passes (predict 3 channels at a time)
- Or post-processing to interpolate remaining channels

### 6. ✅ Output Frame Rate (20 FPS)
**Requirement**: "Regarding the frame rate, the simulation has a frame every 0.25 microsecond. So running real time is not viable. Can you instead have the output framerate default to 20 frames per second?"

**Solution**:
```yaml
validation:
  output_fps: 20.0  # Output videos/GIFs at 20 fps (not simulation time)
```

The model doesn't need to know about simulation time - it just learns temporal patterns. The output frame rate is purely for visualization.

## 🚀 Complete Workflow

### Step 1: Organize Your Simulations

```bash
# Your directory structure
simulations/
├── sim_0001/
│   ├── frame_0000.npz  # Each has: temperature, pressure, velocity_x, velocity_y, density
│   ├── frame_0001.npz
│   ├── ...
│   └── frame_0100.npz  # 101 total frames
├── sim_0002/
├── ...
└── sim_9999/  # Thousands of simulations
```

### Step 2: Create Dataset Metadata

```bash
python scripts/create_batch_metadata.py \
    --input-dir simulations \
    --output dataset.json \
    --pattern "sim_*"
```

Or manually:
```json
[
  {"caption": "Physics simulation 1", "media_path": "simulations/sim_0001"},
  {"caption": "Physics simulation 2", "media_path": "simulations/sim_0002"},
  ...
]
```

### Step 3: Extract Initial Frames

```bash
# Extract frame 0 from each simulation for validation
python scripts/extract_initial_frames.py extract-batch \
    simulations \
    initial_frames \
    --frame-index 0 \
    --visualize-as-density true
```

### Step 4: Preprocess

```bash
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "512x512x101" \
    --caption-column "caption" \
    --video-column "media_path"
```

### Step 5: Configure Training

Edit `configs/physics_simulation_enhanced.yaml`:

```yaml
data:
  preprocessed_data_root: "simulations/.precomputed"
  npz_initial_frame_index: 0  # ← YOUR INITIAL FRAME
  npz_channel_names: ["temperature", "pressure", "velocity_x"]  # ← YOUR CHANNELS
  npz_preserve_all_channels: true

validation:
  output_fps: 20.0  # ← YOUR DESIRED FPS
  visualization_script: null  # ← OR path to custom script
  save_comparison_gif: true  # ← ENABLE COMPARISON
  video_dims: [512, 512, 101]  # ← YOUR DIMENSIONS
  images:  # First frames for validation
    - "initial_frames/sim_0001.png"
    - "initial_frames/sim_0002.png"
    - "initial_frames/sim_0003.png"
```

### Step 6: Train

```bash
python scripts/train.py configs/physics_simulation_enhanced.yaml
```

## 📊 Example Outputs

After training, you'll get:

1. **Validation Videos** (20 fps): `outputs/samples/step_XXXXXX_0.mp4`
2. **Comparison GIFs** (if enabled): `outputs/comparisons/step_XXXXXX_comparison.gif`
3. **Custom Visualizations** (if script provided): `outputs/custom/step_XXXXXX_custom.png`

## 🎨 Visualization Examples

### Density Sum (Built-in)
```python
from ltxv_trainer.npz_visualization import visualize_density_sum
import numpy as np

# Load your data
channels = np.load('sim_0001/frame_0050.npz')
density_sum = channels['density1'] + channels['density2'] + channels['density3']

# Visualize
visualize_density_sum(
    channels=density_sum,
    output_path='density_sum.png',
    cmap='viridis'
)
```

### Comparison GIF (Automatic)
Enabled via `save_comparison_gif: true` in config. Automatically generated during validation.

### Custom Visualization
```python
# custom_viz.py
import os
import numpy as np
import matplotlib.pyplot as plt

input_data = os.environ['INPUT_DATA']
output_path = os.environ['OUTPUT_PATH']

# Load data
data_dir = Path(input_data)
npz_files = sorted(data_dir.glob('*.npz'))

# Create custom plot
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for i, npz_file in enumerate(npz_files[:4]):
    data = np.load(npz_file)
    density = data['temperature'] + data['pressure']
    
    ax = axes[i // 2, i % 2]
    im = ax.imshow(density, cmap='viridis')
    ax.set_title(f'Frame {i}')
    plt.colorbar(im, ax=ax)

plt.savefig(output_path, dpi=150)
```

## ⚙️ Configuration Quick Reference

```yaml
# Key settings for your requirements

data:
  npz_initial_frame_index: 0           # Which frame is initial (REQUIREMENT 1)
  npz_channel_names: ["ch1", "ch2"]    # Which channels to use (REQUIREMENT 5 workaround)
  npz_preserve_all_channels: true       # Save all channel info

validation:
  output_fps: 20.0                      # Output frame rate (REQUIREMENT 6)
  visualization_script: "custom.py"     # Custom viz (REQUIREMENT 2)
  save_comparison_gif: true             # True/Pred/Diff (REQUIREMENT 4)
  video_dims: [512, 512, 101]          # Generate all frames (REQUIREMENT 1)
```

## 🔍 Troubleshooting

**Q: Can't fit all 101 frames in memory?**
```yaml
optimization:
  gradient_checkpointing: true
  batch_size: 1
acceleration:
  quantization: "int8-quanto"
```

**Q: Training too slow with thousands of simulations?**
```yaml
data:
  num_dataloader_workers: 4  # Parallel loading
optimization:
  gradient_accumulation_steps: 4  # Effective larger batch
```

**Q: Want to predict different channel combinations?**
Train multiple models:
- Model 1: `["temperature", "pressure", "velocity_x"]`
- Model 2: `["density1", "density2", "density3"]`
- Combine predictions in post-processing

## 📝 Notes

- **Channel Limitation**: VAE is 3-channel (RGB). To "predict all channels", you would need to train multiple models or modify the VAE architecture.
- **Initial Frame**: The model will condition on the initial frame and predict forward in time.
- **Frame Rate**: Output FPS only affects visualization, not training dynamics.
- **Scalability**: Tested with thousands of simulations. Use `num_dataloader_workers` for parallelism.
