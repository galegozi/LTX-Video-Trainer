# NPZ Physics Simulation Support

This guide covers how to train LTX-Video model with NPZ files from physics simulations using image-to-video mode.

## 📋 Overview

The LTX-Video trainer now supports NPZ files containing physics simulation data. This enables you to:
- Train models on scientific simulation data
- Use image-to-video mode where the first frame conditions the generation
- Predict entire simulation sequences from initial states

## 📁 NPZ File Format

Your NPZ files must follow this structure:

```python
import numpy as np

# Create example NPZ file
frames = np.random.rand(121, 512, 512, 3)  # (T, H, W, C) format
fps = 24  # Optional, defaults to 24 if not provided

np.savez('simulation_001.npz', frames=frames, fps=fps)
```

### Required Keys

- **`frames`** or **`data`**: Numpy array containing the simulation frames
  - Supported shapes:
    - `(T, H, W, C)` - Time, Height, Width, Channels (will be converted to `(T, C, H, W)`)
    - `(T, C, H, W)` - Time, Channels, Height, Width (preferred format)
    - `(T, H, W)` - Grayscale without channel dimension (will be expanded to 3 channels)
  - Channels: 1 (grayscale, will be expanded), 3 (RGB), or 4 (RGBA, alpha will be dropped)
  - Value range: `[0, 1]` (normalized) or `[0, 255]` (will be automatically normalized)

### Optional Keys

- **`fps`**: Frames per second (float). Defaults to 24 if not provided.

## 🎬 Dataset Preparation

### Step 1: Prepare Your Dataset Metadata

Create a JSON, JSONL, or CSV file that references your NPZ files:

**JSON format:**
```json
[
  {
    "caption": "Fluid dynamics simulation with high viscosity",
    "media_path": "simulations/fluid_001.npz"
  },
  {
    "caption": "Particle collision at high energy",
    "media_path": "simulations/particle_002.npz"
  }
]
```

**JSONL format:**
```jsonl
{"caption": "Fluid dynamics simulation with high viscosity", "media_path": "simulations/fluid_001.npz"}
{"caption": "Particle collision at high energy", "media_path": "simulations/particle_002.npz"}
```

**CSV format:**
```csv
caption,media_path
"Fluid dynamics simulation with high viscosity","simulations/fluid_001.npz"
"Particle collision at high energy","simulations/particle_002.npz"
```

### Step 2: Preprocess Your NPZ Dataset

Use the standard preprocessing script to process your NPZ files:

```bash
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "512x512x121" \
    --caption-column "caption" \
    --video-column "media_path" \
    --model-source "LTXV_13B_097_DEV"
```

The preprocessing script will:
1. Load each NPZ file
2. Extract and normalize frames
3. Resize and crop to match the resolution bucket
4. Compute VAE latents
5. Compute text embeddings for captions

### Step 3: Configure Training for Image-to-Video

Create a training configuration that uses first-frame conditioning:

```yaml
# Model configuration
model:
  model_source: "LTXV_13B_097_DEV"
  training_mode: "lora"

# LoRA configuration
lora:
  rank: 128
  alpha: 128

# Conditioning configuration - IMPORTANT for image-to-video
conditioning:
  mode: "none"
  first_frame_conditioning_p: 1.0  # Always condition on first frame

# Data configuration
data:
  preprocessed_data_root: "/path/to/preprocessed/data"
  num_dataloader_workers: 2

# Validation configuration
validation:
  prompts:
    - "Fluid dynamics simulation with high viscosity"
    - "Particle collision at high energy"
  # Extract first frames from NPZ files for validation
  images:
    - "first_frames/fluid_001.png"
    - "first_frames/particle_002.png"
  video_dims: [512, 512, 121]
  seed: 42
  inference_steps: 50
  interval: 250

# Other settings...
optimization:
  learning_rate: 2e-4
  steps: 2000
  batch_size: 1
  gradient_accumulation_steps: 1

output_dir: "outputs/physics_simulation_lora"
```

### Step 4: Extract First Frames for Validation (Optional)

To use image-to-video mode during validation, extract the first frame from each NPZ file:

```python
import numpy as np
from PIL import Image

def extract_first_frame(npz_path, output_path):
    """Extract and save the first frame from an NPZ file."""
    data = np.load(npz_path)
    frames = data['frames'] if 'frames' in data else data['data']
    
    # Get first frame
    first_frame = frames[0]
    
    # Handle different formats
    if first_frame.ndim == 3 and first_frame.shape[-1] in [1, 3, 4]:
        # (H, W, C) format
        pass
    elif first_frame.ndim == 3:
        # (C, H, W) format - convert to (H, W, C)
        first_frame = np.transpose(first_frame, (1, 2, 0))
    elif first_frame.ndim == 2:
        # (H, W) grayscale - expand to RGB
        first_frame = np.stack([first_frame] * 3, axis=-1)
    
    # Normalize to [0, 255] if needed
    if first_frame.max() <= 1.0:
        first_frame = (first_frame * 255).astype(np.uint8)
    else:
        first_frame = first_frame.astype(np.uint8)
    
    # Save as PNG
    if first_frame.shape[-1] == 4:
        # RGBA
        img = Image.fromarray(first_frame, mode='RGBA')
    elif first_frame.shape[-1] == 3:
        # RGB
        img = Image.fromarray(first_frame, mode='RGB')
    else:
        # Grayscale
        img = Image.fromarray(first_frame[:, :, 0], mode='L')
    
    img.save(output_path)

# Example usage
extract_first_frame('simulations/fluid_001.npz', 'first_frames/fluid_001.png')
```

### Step 5: Train Your Model

```bash
python scripts/train.py configs/physics_simulation_config.yaml
```

## 🎯 Training Modes

### Image-to-Video (Recommended for Physics Simulations)

Configure your model to always condition on the first frame:

```yaml
conditioning:
  mode: "none"
  first_frame_conditioning_p: 1.0  # 100% probability of first-frame conditioning
```

This approach is ideal because:
- The initial simulation state (first frame) conditions the entire prediction
- The model learns to predict temporal evolution from initial conditions
- More stable and predictable results

### Alternative: Standard Video Training

For comparison or experimentation, you can train without first-frame conditioning:

```yaml
conditioning:
  mode: "none"
  first_frame_conditioning_p: 0.0  # No first-frame conditioning
```

This approach:
- Trains on full sequences without conditioning
- May be useful for style transfer or general patterns
- Less suitable for physics prediction tasks

## 📊 Best Practices

### Resolution Guidelines

- **High detail simulations**: Use larger spatial dimensions (768×768) with fewer frames (25-49)
- **Long temporal sequences**: Use smaller spatial dimensions (512×512) with more frames (121-161)
- **Memory constraints**: Balance spatial and temporal dimensions based on available GPU memory

### Frame Count

Remember that frame count must follow the formula: `frames = 8n + 1` (e.g., 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121, 161)

### Captions for Physics Simulations

Good captions should describe:
- Physical phenomena (e.g., "turbulent flow", "wave propagation")
- Key parameters (e.g., "high viscosity", "low Reynolds number")
- Observable features (e.g., "vortex formation", "collision dynamics")

Example captions:
```
"Navier-Stokes simulation showing turbulent flow around cylinder"
"Wave equation propagation with reflective boundaries"
"Molecular dynamics of gas particles at high temperature"
"Finite element analysis showing stress distribution in beam"
```

## 🔍 Debugging and Validation

### Verify NPZ Files

```python
import numpy as np

# Load and inspect NPZ file
data = np.load('simulation_001.npz')
print("Keys:", list(data.keys()))
print("Frames shape:", data['frames'].shape)
print("Value range:", data['frames'].min(), "-", data['frames'].max())
print("FPS:", data.get('fps', 'Not specified'))
```

### Decode Preprocessed Latents

Use the `--decode-videos` flag to verify preprocessing:

```bash
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "512x512x121" \
    --decode-videos
```

This will save decoded videos in `.precomputed/decoded_videos/` for visual inspection.

## 🚀 Example: Complete Workflow

Here's a complete example workflow:

```bash
# 1. Create dataset metadata
cat > physics_dataset.json << EOF
[
  {"caption": "Fluid simulation", "media_path": "sims/fluid_001.npz"},
  {"caption": "Wave propagation", "media_path": "sims/wave_001.npz"}
]
EOF

# 2. Preprocess dataset
python scripts/preprocess_dataset.py physics_dataset.json \
    --resolution-buckets "512x512x121" \
    --caption-column "caption" \
    --video-column "media_path" \
    --decode-videos

# 3. Extract first frames for validation (Python script)
python extract_first_frames.py

# 4. Train model
python scripts/train.py configs/physics_lora.yaml
```

## 💡 Tips

- **Data Quality**: Ensure your NPZ files contain consistent, high-quality simulation data
- **Normalization**: The preprocessor handles normalization, but verify your input ranges
- **Frame Count**: Use consistent frame counts across your dataset for best results
- **Validation**: Always use first-frame conditioning during validation for physics predictions
- **Computational Cost**: Physics simulations can be memory-intensive; adjust batch size accordingly

## 🤝 Need Help?

If you encounter issues:
1. Verify your NPZ file format matches the specification
2. Check preprocessing logs for warnings or errors
3. Try decoding videos to verify preprocessing worked correctly
4. Join our [Discord community](https://discord.gg/Mn8BRgUKKy) for support

Happy training with physics simulations! 🎉
