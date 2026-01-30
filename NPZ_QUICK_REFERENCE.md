# NPZ Physics Simulation Support - Quick Reference

## Summary

This implementation adds comprehensive support for training LTX-Video models with NPZ files from physics simulations using image-to-video mode.

## Supported NPZ Formats

### 1. Single NPZ with All Frames
```python
np.savez('sim.npz', frames=np.array([T, H, W, C]), fps=24)
```

### 2. Single NPZ with Multiple Channels (Per-Frame)
```python
np.savez('frame_0000.npz',
    temperature=np.array([H, W]),
    pressure=np.array([H, W]),
    channels=['temperature', 'pressure', 'velocity_x'],
    fps=24)
```

### 3. Directory of NPZ Files (Recommended for Per-Frame Data)
```
simulation_001/
├── frame_0000.npz
├── frame_0001.npz
├── frame_0002.npz
└── ...
```

## Quick Start

### Create Example Data
```bash
python scripts/create_npz_sequence.py create-example \
    --output-dir my_simulations \
    --num-simulations 3 \
    --num-frames 25 \
    --resolution 256
```

### Preprocess Dataset
```bash
python scripts/preprocess_dataset.py my_simulations/dataset.json \
    --resolution-buckets "256x256x25" \
    --caption-column "caption" \
    --video-column "media_path"
```

### Train Model
```bash
python scripts/train.py configs/physics_simulation_npz.yaml
```

## Utility Scripts

### Create NPZ Sequence
```bash
scripts/create_npz_sequence.py create-example [OPTIONS]
```

### Extract First Frames
```bash
scripts/npz_utils.py extract-frames INPUT_DIR OUTPUT_DIR
```

### Validate NPZ Format
```bash
scripts/npz_utils.py validate NPZ_FILE
```

### Create Example NPZ
```bash
scripts/npz_utils.py create-example OUTPUT_PATH --frames 25 --width 512 --height 512
```

## Key Features

✅ Supports three different NPZ formats  
✅ Handles multiple simulation channels (temperature, pressure, velocity, etc.)  
✅ Automatic normalization of channel values  
✅ Channel selection and RGB mapping  
✅ Directory-based NPZ sequences (one file per frame)  
✅ Image-to-video conditioning with first frame  
✅ Comprehensive validation and testing  
✅ Helper scripts for data preparation  

## Dataset Structure

```
dataset/
├── dataset.json              # Metadata file
├── simulation_001/           # NPZ sequence directory
│   ├── frame_0000.npz
│   ├── frame_0001.npz
│   └── ...
├── simulation_002/
│   ├── frame_0000.npz
│   └── ...
└── .precomputed/            # Created after preprocessing
    ├── latents/
    └── conditions/
```

## Configuration

Key configuration for image-to-video physics training:

```yaml
conditioning:
  mode: "none"
  first_frame_conditioning_p: 1.0  # Always condition on first frame

validation:
  images:  # Provide first frames for validation
    - "first_frames/sim_001.png"
```

## Documentation

- 📖 Full Guide: `docs/npz-physics-simulations.md`
- 🔧 Utility Scripts: `scripts/npz_utils.py`, `scripts/create_npz_sequence.py`
- ⚙️ Config Example: `configs/physics_simulation_npz.yaml`
- 🧪 Tests: `scripts/test_npz_support.py`

## Example Workflow

1. **Prepare simulation data** → NPZ sequences
2. **Create dataset metadata** → JSON/CSV file
3. **Preprocess** → Compute latents and text embeddings
4. **Train** → Image-to-video LoRA training
5. **Validate** → Generate predictions from initial states

## Requirements

- NumPy for NPZ file operations
- PyTorch for tensor operations
- Standard LTX-Video trainer dependencies

## Testing

```bash
# Run NPZ format tests
python scripts/test_npz_support.py

# Create and validate example data
python scripts/create_npz_sequence.py create-example --output-dir /tmp/test
python scripts/npz_utils.py validate /tmp/test/simulation_001/frame_0000.npz
```

## Notes

- Frame counts must follow the formula: `frames = 8n + 1` (e.g., 9, 17, 25, 33, 41, 49, 81, 121)
- NPZ files in a sequence directory are loaded in alphabetical order
- The first 3 channels are mapped to RGB for visualization
- Values are automatically normalized to [0, 1] range
- Supports various channel formats: grayscale, RGB, RGBA, multi-channel

## Support

For issues or questions:
- Check documentation: `docs/npz-physics-simulations.md`
- Run validation: `scripts/npz_utils.py validate YOUR_FILE.npz`
- Join Discord: https://discord.gg/Mn8BRgUKKy
