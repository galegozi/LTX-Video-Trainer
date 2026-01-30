"""
Multi-channel VAE wrapper for handling N-channel video data.

This module provides a wrapper around the standard 3-channel LTX-Video VAE
to support encoding and decoding of videos with arbitrary number of channels.

Strategy: Channel Grouping
- Split N channels into groups of 3
- Encode each group separately
- Concatenate latents along channel dimension
- Decode groups separately and recombine
"""

import torch
from torch import Tensor
from diffusers import AutoencoderKLLTXVideo
from typing import Optional, Tuple
import math

from ltxv_trainer import logger


class MultiChannelVAEWrapper:
    """Wrapper to handle multi-channel video encoding/decoding.
    
    This class enables the 3-channel LTX-Video VAE to process videos with
    arbitrary number of channels by grouping channels into sets of 3,
    processing each group independently, and combining the results.
    
    Example:
        For 10 channels:
        - Group 1: channels [0, 1, 2]
        - Group 2: channels [3, 4, 5]
        - Group 3: channels [6, 7, 8]
        - Group 4: channels [9, 9, 9] (last channel repeated)
        
        Each group is encoded separately, and latents are concatenated.
        During decoding, latents are split and decoded, then channels recombined.
    """
    
    def __init__(
        self,
        vae: AutoencoderKLLTXVideo,
        num_channels: int = 3,
        channel_grouping_strategy: str = "sequential",
    ):
        """Initialize multi-channel VAE wrapper.
        
        Args:
            vae: Base 3-channel VAE model
            num_channels: Total number of channels to support
            channel_grouping_strategy: How to group channels ("sequential" or "interleaved")
        """
        self.vae = vae
        self.num_channels = num_channels
        self.channel_grouping_strategy = channel_grouping_strategy
        
        # Calculate number of groups needed (ceil division)
        self.num_groups = math.ceil(num_channels / 3)
        
        # Determine which channels go into which group
        self._setup_channel_groups()
        
        logger.info(
            f"MultiChannelVAEWrapper initialized: {num_channels} channels "
            f"split into {self.num_groups} groups of 3"
        )
    
    def _setup_channel_groups(self) -> None:
        """Setup channel group assignments."""
        self.channel_groups = []
        
        if self.channel_grouping_strategy == "sequential":
            # Sequential: [0,1,2], [3,4,5], [6,7,8], etc.
            for group_idx in range(self.num_groups):
                start_ch = group_idx * 3
                end_ch = min(start_ch + 3, self.num_channels)
                
                # Get actual channel indices
                channels = list(range(start_ch, end_ch))
                
                # Pad with last channel if needed
                while len(channels) < 3:
                    channels.append(channels[-1])
                
                self.channel_groups.append(channels)
        
        elif self.channel_grouping_strategy == "interleaved":
            # Interleaved: distribute channels evenly across groups
            # [0,3,6], [1,4,7], [2,5,8], etc.
            for group_idx in range(self.num_groups):
                channels = []
                for i in range(3):
                    ch_idx = group_idx + i * self.num_groups
                    if ch_idx < self.num_channels:
                        channels.append(ch_idx)
                    else:
                        # Pad with last valid channel
                        channels.append(channels[-1] if channels else 0)
                
                self.channel_groups.append(channels)
        
        else:
            raise ValueError(f"Unknown channel grouping strategy: {self.channel_grouping_strategy}")
        
        logger.debug(f"Channel groups: {self.channel_groups}")
    
    def encode(
        self,
        video: Tensor,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[Tensor, dict]:
        """Encode multi-channel video to latents.
        
        Args:
            video: Input video tensor of shape [B, F, C, H, W] where C = num_channels
            generator: Random generator for sampling
            
        Returns:
            Tuple of (combined_latents, metadata)
            - combined_latents: Shape [B, 128*num_groups, F', H', W']
            - metadata: Dict with channel grouping info
        """
        batch_size, num_frames, channels, height, width = video.shape
        
        if channels != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {channels}"
            )
        
        # Encode each channel group
        group_latents = []
        for group_idx, channel_indices in enumerate(self.channel_groups):
            # Extract channels for this group
            group_video = video[:, :, channel_indices, :, :]  # [B, F, 3, H, W]
            
            # Encode with VAE
            latent_dist = self.vae.encode(group_video)
            group_latent = latent_dist.latent_dist.sample(generator=generator)
            
            group_latents.append(group_latent)
        
        # Combine latents by concatenating along channel dimension
        # Each group_latent is [B, 128, F', H', W']
        combined_latents = torch.cat(group_latents, dim=1)  # [B, 128*num_groups, F', H', W']
        
        metadata = {
            'num_channels': self.num_channels,
            'num_groups': self.num_groups,
            'channel_groups': self.channel_groups,
            'original_shape': video.shape,
        }
        
        return combined_latents, metadata
    
    def decode(
        self,
        latents: Tensor,
        metadata: Optional[dict] = None,
        decode_timestep: float = 0.0,
        decode_noise_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """Decode latents to multi-channel video.
        
        Args:
            latents: Combined latents of shape [B, 128*num_groups, F', H', W']
            metadata: Channel grouping metadata from encoding
            decode_timestep: Timestep for decoding
            decode_noise_scale: Noise scale for decoding
            generator: Random generator
            
        Returns:
            Reconstructed video of shape [B, F, num_channels, H, W]
        """
        # Validate metadata
        if metadata is not None:
            if metadata['num_channels'] != self.num_channels:
                logger.warning(
                    f"Metadata num_channels ({metadata['num_channels']}) "
                    f"doesn't match wrapper num_channels ({self.num_channels})"
                )
        
        # Split latents by group
        latent_groups = torch.chunk(latents, self.num_groups, dim=1)
        
        # Decode each group
        group_videos = []
        for group_idx, group_latent in enumerate(latent_groups):
            # Decode group
            decoded = self.vae.decode(
                group_latent,
                timestep=decode_timestep,
                noise_scale=decode_noise_scale,
                generator=generator
            )
            
            # decoded is [B, F, 3, H, W]
            group_videos.append(decoded)
        
        # Recombine channels
        # Stack all decoded groups: [B, F, 3*num_groups, H, W]
        all_channels = torch.cat(group_videos, dim=2)
        
        # Extract only the original channels (remove padding)
        # Build channel extraction indices
        original_channel_indices = []
        for group_idx, channel_indices in enumerate(self.channel_groups):
            for local_ch_idx, global_ch_idx in enumerate(channel_indices):
                if global_ch_idx < self.num_channels and global_ch_idx not in original_channel_indices:
                    # Map to position in concatenated tensor
                    concat_idx = group_idx * 3 + local_ch_idx
                    original_channel_indices.append((global_ch_idx, concat_idx))
        
        # Sort by global channel index
        original_channel_indices.sort(key=lambda x: x[0])
        
        # Extract channels in correct order
        output_channels = []
        for global_ch_idx, concat_idx in original_channel_indices:
            if global_ch_idx < self.num_channels:
                output_channels.append(all_channels[:, :, concat_idx:concat_idx+1, :, :])
        
        # Concatenate to get final output
        output_video = torch.cat(output_channels, dim=2)  # [B, F, num_channels, H, W]
        
        return output_video
    
    @property
    def device(self):
        """Get device of underlying VAE."""
        return self.vae.device
    
    @property
    def dtype(self):
        """Get dtype of underlying VAE."""
        return self.vae.dtype
    
    @property
    def latents_mean(self):
        """Get latents mean from underlying VAE."""
        return self.vae.latents_mean
    
    @property
    def latents_std(self):
        """Get latents std from underlying VAE."""
        return self.vae.latents_std
    
    def to(self, *args, **kwargs):
        """Move VAE to device/dtype."""
        self.vae = self.vae.to(*args, **kwargs)
        return self
    
    def enable_tiling(self):
        """Enable tiling on underlying VAE."""
        self.vae.enable_tiling()
    
    def disable_tiling(self):
        """Disable tiling on underlying VAE."""
        self.vae.disable_tiling()


def create_multi_channel_vae(
    vae: AutoencoderKLLTXVideo,
    num_channels: int,
    channel_grouping_strategy: str = "sequential"
) -> MultiChannelVAEWrapper | AutoencoderKLLTXVideo:
    """Factory function to create appropriate VAE wrapper.
    
    Args:
        vae: Base VAE model
        num_channels: Number of channels needed
        channel_grouping_strategy: Strategy for grouping channels
        
    Returns:
        MultiChannelVAEWrapper if num_channels != 3, else original VAE
    """
    if num_channels == 3:
        logger.info("Using standard 3-channel VAE (no wrapper needed)")
        return vae
    
    logger.info(f"Creating multi-channel VAE wrapper for {num_channels} channels")
    return MultiChannelVAEWrapper(
        vae=vae,
        num_channels=num_channels,
        channel_grouping_strategy=channel_grouping_strategy
    )
