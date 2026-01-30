"""
Multi-channel encoding/decoding utilities.

Extends the standard ltxv_utils to support multi-channel video encoding and decoding.
"""

import torch
from torch import Tensor
from typing import Optional

from ltxv_trainer.multi_channel_vae import MultiChannelVAEWrapper
from ltxv_trainer.ltxv_utils import pack_latents, unpack_latents, _normalize_latents, _unnormalize_latents
from ltxv_trainer import logger


def encode_multi_channel_video(
    vae_wrapper: MultiChannelVAEWrapper,
    image_or_video: Tensor,
    patch_size: int = 1,
    patch_size_t: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """Encode multi-channel video to latents using MultiChannelVAEWrapper.
    
    Args:
        vae_wrapper: Multi-channel VAE wrapper
        image_or_video: Input tensor of shape [B,C,F,H,W] where C = num_channels
        patch_size: Spatial patch size
        patch_size_t: Temporal patch size
        device: Target device
        dtype: Target dtype
        generator: Random generator
        
    Returns:
        Dict containing latents and metadata
    """
    device = device or vae_wrapper.device
    
    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D"
    
    batch_size, num_channels, num_frames, height, width = image_or_video.shape
    
    # Move to device and dtype
    image_or_video = image_or_video.to(device=device, dtype=vae_wrapper.dtype)
    
    # Permute to [B, F, C, H, W] for VAE
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()
    
    # Encode with multi-channel wrapper
    logger.info(f"Encoding multi-channel video: shape {image_or_video.shape}")
    latents, channel_metadata = vae_wrapper.encode(image_or_video, generator=generator)
    latents = latents.to(dtype=dtype)
    
    _, latent_channels, latent_frames, latent_height, latent_width = latents.shape
    
    # Normalize latents
    # Note: Using base VAE's mean/std for now, may need per-group normalization
    latents = _normalize_latents(latents, vae_wrapper.latents_mean, vae_wrapper.latents_std)
    
    # Pack latents
    latents_packed = pack_latents(latents, patch_size, patch_size_t)
    
    return {
        'latents': latents_packed,
        'num_frames': latent_frames,
        'height': latent_height,
        'width': latent_width,
        'channel_metadata': channel_metadata,
        'num_groups': vae_wrapper.num_groups,
        'original_num_channels': num_channels,
    }


def decode_multi_channel_video(
    vae_wrapper: MultiChannelVAEWrapper,
    latents: Tensor,
    num_frames: int,
    height: int,
    width: int,
    channel_metadata: Optional[dict] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    patch_size: int = 1,
    patch_size_t: int = 1,
    decode_timestep: float = 0.0,
    decode_noise_scale: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Decode latents to multi-channel video using MultiChannelVAEWrapper.
    
    Args:
        vae_wrapper: Multi-channel VAE wrapper
        latents: Packed latents from encoding
        num_frames: Number of latent frames
        height: Latent height
        width: Latent width
        channel_metadata: Channel grouping metadata from encoding
        device: Target device
        dtype: Target dtype
        patch_size: Spatial patch size
        patch_size_t: Temporal patch size
        decode_timestep: Timestep for decoding
        decode_noise_scale: Noise scale
        generator: Random generator
        
    Returns:
        Decoded multi-channel video [B, F, C, H, W]
    """
    device = device or vae_wrapper.device
    
    # Unpack latents
    latents_unpacked = unpack_latents(
        latents,
        num_frames,
        height,
        width,
        vae_wrapper.num_groups * 128,  # Latent channels = 128 * num_groups
        patch_size,
        patch_size_t
    )
    
    # Unnormalize latents
    latents_unnormalized = _unnormalize_latents(
        latents_unpacked,
        vae_wrapper.latents_mean,
        vae_wrapper.latents_std
    )
    
    # Move to device
    latents_unnormalized = latents_unnormalized.to(device=device, dtype=vae_wrapper.dtype)
    
    # Decode with multi-channel wrapper
    logger.info(f"Decoding multi-channel latents: shape {latents_unnormalized.shape}")
    decoded_video = vae_wrapper.decode(
        latents_unnormalized,
        metadata=channel_metadata,
        decode_timestep=decode_timestep,
        decode_noise_scale=decode_noise_scale,
        generator=generator
    )
    
    # decoded_video is [B, F, C, H, W]
    # Permute back to [B, C, F, H, W]
    decoded_video = decoded_video.permute(0, 2, 1, 3, 4).contiguous()
    
    if dtype is not None:
        decoded_video = decoded_video.to(dtype=dtype)
    
    return decoded_video
