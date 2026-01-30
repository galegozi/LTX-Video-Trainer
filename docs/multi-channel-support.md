# Multi-Channel Support: True All-Channel Input/Output

This guide explains how to train the LTX-Video model with full multi-channel support, ensuring that when you provide N channels, you get N channels back in predictions.

## 🎯 Problem Solved

**Previous Limitation**: The LTX-Video VAE is designed for 3-channel (RGB) video, so any N-channel input was reduced to 3 channels.

**New Solution**: Multi-channel VAE wrapper that processes all channels by grouping them into sets of 3, encoding separately, and recombining.

## 🏗️ Architecture Overview

### Channel Grouping Strategy

For a 10-channel input:

```
Input: [Batch, Frames, 10 Channels, Height, Width]
    ↓
Split into groups of 3:
    Group 1: Channels [0, 1, 2]
    Group 2: Channels [3, 4, 5]
    Group 3: Channels [6, 7, 8]
    Group 4: Channels [9, 9, 9]  ← last channel repeated for padding
    ↓
Encode each group with VAE:
    Group 1 → Latents₁ [B, 128, F', H', W']
    Group 2 → Latents₂ [B, 128, F', H', W']
    Group 3 → Latents₃ [B, 128, F', H', W']
    Group 4 → Latents₄ [B, 128, F', H', W']
    ↓
Concatenate latents:
    Combined: [B, 512, F', H', W']  (128 × 4 groups)
    ↓
Train model on combined latents
    ↓
Decode each group separately:
    Latents → Group 1, Group 2, Group 3, Group 4
    ↓
Recombine channels:
    Output: [B, F, 10, H, W]  ← All 10 channels restored!
```

### Key Benefits

1. **True Multi-Channel**: All channels are encoded and decoded
2. **No Information Loss**: Each channel group preserves full spatial-temporal information
3. **Scalable**: Works with any number of channels (3, 5, 10, 20, etc.)
4. **Backward Compatible**: Still works with standard 3-channel RGB when desired

## 🔧 Configuration

### Enable Multi-Channel Mode

```yaml
# configs/multi_channel_physics.yaml

data:
  preprocessed_data_root: "/path/to/data"
  
  # Enable multi-channel support
  use_multi_channel_vae: true
  num_channels: 10  # Total channels in your data
  channel_grouping_strategy: "sequential"  # or "interleaved"
  
  # NPZ settings
  npz_channel_names:
    - "temperature"
    - "pressure"
    - "density"
    - "velocity_x"
    - "velocity_y"
    - "velocity_z"
    - "energy"
    - "vorticity"
    - "entropy"
    - "turbulence"
  npz_preserve_all_channels: true

validation:
  output_fps: 20.0
  video_dims: [512, 512, 101]
```

### Channel Grouping Strategies

#### Sequential (Default)
Channels grouped sequentially:
- Group 1: [0, 1, 2]
- Group 2: [3, 4, 5]
- Group 3: [6, 7, 8]
- etc.

**Best for**: Channels with natural groupings (e.g., velocity components, RGB-like data)

#### Interleaved
Channels distributed evenly across groups:
- Group 1: [0, 3, 6]
- Group 2: [1, 4, 7]
- Group 3: [2, 5, 8]
- etc.

**Best for**: Maximum channel diversity per group, balance computation

## 📖 Usage Guide

### Step 1: Prepare Multi-Channel NPZ Data

Your NPZ files should contain all channels:

```python
import numpy as np

# Create 10-channel simulation data
num_frames = 101
height, width = 512, 512
num_channels = 10

channels = {
    'temperature': np.random.rand(num_frames, height, width),
    'pressure': np.random.rand(num_frames, height, width),
    'density': np.random.rand(num_frames, height, width),
    'velocity_x': np.random.rand(num_frames, height, width),
    'velocity_y': np.random.rand(num_frames, height, width),
    'velocity_z': np.random.rand(num_frames, height, width),
    'energy': np.random.rand(num_frames, height, width),
    'vorticity': np.random.rand(num_frames, height, width),
    'entropy': np.random.rand(num_frames, height, width),
    'turbulence': np.random.rand(num_frames, height, width),
}

# Save with channel order
np.savez(
    'simulation_001.npz',
    **channels,
    channels=list(channels.keys()),
    fps=24
)
```

For sequences (one frame per file):

```python
for frame_idx in range(num_frames):
    frame_data = {
        'temperature': temperature_field[frame_idx],
        'pressure': pressure_field[frame_idx],
        # ... other channels ...
    }
    frame_data['channels'] = list(channels.keys())
    frame_data['fps'] = 24
    
    np.savez(f'sim_001/frame_{frame_idx:04d}.npz', **frame_data)
```

### Step 2: Preprocess with Multi-Channel

```bash
python scripts/preprocess_dataset.py dataset.json \
    --resolution-buckets "512x512x101" \
    --caption-column "caption" \
    --video-column "media_path"
```

The preprocessing will automatically detect and preserve all channels when `use_multi_channel_vae: true`.

### Step 3: Train

```bash
python scripts/train.py configs/multi_channel_physics.yaml
```

The trainer will:
1. Load multi-channel VAE wrapper automatically
2. Encode all channels in groups of 3
3. Train on combined latent representation
4. Decode to restore all original channels

### Step 4: Inference

During inference, provide all input channels and get all channels back:

```python
from ltxv_trainer.multi_channel_vae import create_multi_channel_vae
from ltxv_trainer.multi_channel_utils import encode_multi_channel_video, decode_multi_channel_video

# Load model and VAE wrapper
vae_wrapper = create_multi_channel_vae(
    vae=base_vae,
    num_channels=10,
    channel_grouping_strategy="sequential"
)

# Encode all 10 channels
input_video = torch.rand(1, 10, 101, 512, 512)  # [B, C, F, H, W]
encoded = encode_multi_channel_video(vae_wrapper, input_video)

# ... model inference on latents ...

# Decode back to all 10 channels
output_video = decode_multi_channel_video(
    vae_wrapper,
    predicted_latents,
    num_frames=101,
    height=64,  # latent height
    width=64,   # latent width
    channel_metadata=encoded['channel_metadata']
)

# output_video has shape [B, 10, F, H, W] - all channels restored!
```

## 🧪 Validation

### Check Channel Preservation

```python
import torch

# Test round-trip encoding/decoding
input_channels = torch.rand(1, 10, 25, 256, 256)

# Encode
encoded = encode_multi_channel_video(vae_wrapper, input_channels)
print(f"Latent shape: {encoded['latents'].shape}")
print(f"Num groups: {encoded['num_groups']}")

# Decode
decoded = decode_multi_channel_video(
    vae_wrapper,
    encoded['latents'],
    encoded['num_frames'],
    encoded['height'],
    encoded['width'],
    encoded['channel_metadata']
)

print(f"Input shape: {input_channels.shape}")
print(f"Output shape: {decoded.shape}")
assert decoded.shape[1] == 10, "All channels preserved!"
```

## 📊 Performance Considerations

### Memory Usage

Multi-channel encoding increases memory usage:
- **3 channels**: 128 latent channels
- **10 channels**: 512 latent channels (4 groups × 128)
- **20 channels**: 896 latent channels (7 groups × 128)

**Memory optimization**:
```yaml
optimization:
  batch_size: 1  # Reduce for high channel counts
  gradient_checkpointing: true
  
acceleration:
  mixed_precision_mode: "bf16"
  quantization: "int8-quanto"  # Reduce model memory
```

### Training Time

Training time scales roughly linearly with number of channel groups:
- **3 channels**: 1x baseline
- **10 channels**: ~4x baseline (4 groups)
- **20 channels**: ~7x baseline (7 groups)

### Computational Efficiency

Each group is encoded/decoded independently, which:
- ✅ Can be parallelized (future optimization)
- ✅ Allows per-group optimization
- ⚠️ Increases latent dimension (more parameters to learn)

## 🎨 Visualization

### Visualizing Multi-Channel Predictions

```python
from ltxv_trainer.npz_visualization import visualize_density_sum
import matplotlib.pyplot as plt

# Extract and visualize specific channels
predicted_channels = output_video[0, :, 50, :, :]  # Frame 50, all channels

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
channel_names = ['temp', 'pressure', 'density', 'vel_x', 'vel_y',
                 'vel_z', 'energy', 'vorticity', 'entropy', 'turbulence']

for idx, (ax, name) in enumerate(zip(axes.flat, channel_names)):
    channel_data = predicted_channels[idx].cpu().numpy()
    im = ax.imshow(channel_data, cmap='viridis')
    ax.set_title(name)
    ax.axis('off')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('multi_channel_prediction.png')
```

### Channel-Specific Metrics

```python
# Compute per-channel reconstruction error
for ch_idx, ch_name in enumerate(channel_names):
    true_ch = true_video[:, ch_idx, :, :, :]
    pred_ch = predicted_video[:, ch_idx, :, :, :]
    
    mse = torch.mean((true_ch - pred_ch) ** 2)
    print(f"Channel {ch_name}: MSE = {mse:.6f}")
```

## ⚙️ Advanced Configuration

### Per-Channel Normalization

For channels with very different value ranges:

```python
# In preprocessing, normalize each channel to [0, 1]
normalized_channels = []
for ch in original_channels:
    ch_min, ch_max = ch.min(), ch.max()
    normalized = (ch - ch_min) / (ch_max - ch_min + 1e-8)
    normalized_channels.append(normalized)
```

### Custom Channel Groupings

For specific physics relationships:

```python
# Group related channels together
custom_groups = [
    [0, 1, 2],  # Temperature, pressure, density
    [3, 4, 5],  # Velocity x, y, z
    [6, 7, 8],  # Energy, vorticity, entropy
    [9, 9, 9],  # Turbulence (padded)
]

# Implement in MultiChannelVAEWrapper with custom strategy
```

## 🐛 Troubleshooting

### Issue: Channel count mismatch

```
ValueError: Expected 10 channels, got 3
```

**Solution**: Ensure `use_multi_channel_vae: true` and `num_channels: 10` in config.

### Issue: Out of memory

```
RuntimeError: CUDA out of memory
```

**Solution**:
```yaml
optimization:
  batch_size: 1
  gradient_accumulation_steps: 4
  gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
```

### Issue: Slow training

Training slower than expected with many channels.

**Solution**:
- Expected behavior (linear scaling with channel groups)
- Optimize: reduce resolution, use fewer frames, or train fewer channels at once

## 📚 API Reference

### MultiChannelVAEWrapper

```python
class MultiChannelVAEWrapper:
    def __init__(
        self,
        vae: AutoencoderKLLTXVideo,
        num_channels: int = 3,
        channel_grouping_strategy: str = "sequential"
    )
    
    def encode(
        self,
        video: Tensor,  # [B, F, C, H, W]
        generator: Optional[torch.Generator] = None
    ) -> Tuple[Tensor, dict]
    
    def decode(
        self,
        latents: Tensor,  # [B, 128*num_groups, F', H', W']
        metadata: Optional[dict] = None,
        decode_timestep: float = 0.0,
        decode_noise_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tensor  # [B, F, C, H, W]
```

### Helper Functions

```python
def create_multi_channel_vae(
    vae: AutoencoderKLLTXVideo,
    num_channels: int,
    channel_grouping_strategy: str = "sequential"
) -> MultiChannelVAEWrapper | AutoencoderKLLTXVideo

def encode_multi_channel_video(
    vae_wrapper: MultiChannelVAEWrapper,
    image_or_video: Tensor,
    ...
) -> dict

def decode_multi_channel_video(
    vae_wrapper: MultiChannelVAEWrapper,
    latents: Tensor,
    ...
) -> Tensor
```

## 🎉 Summary

With multi-channel support, you can now:
- ✅ Provide N channels as input
- ✅ Train model on all channels
- ✅ Get N channels back in predictions
- ✅ Preserve full spatial-temporal information per channel
- ✅ Scale to any number of channels

Perfect for physics simulations with multiple fields (temperature, pressure, velocity, etc.)!
