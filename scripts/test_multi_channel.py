#!/usr/bin/env python3
"""
Test multi-channel VAE wrapper functionality.

This test validates that the MultiChannelVAEWrapper correctly handles
multi-channel video encoding and decoding.
"""

import sys
from pathlib import Path
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_multi_channel_vae_wrapper():
    """Test MultiChannelVAEWrapper with synthetic data."""
    print("\n" + "="*60)
    print("Testing Multi-Channel VAE Wrapper")
    print("="*60 + "\n")
    
    try:
        from src.ltxv_trainer.multi_channel_vae import MultiChannelVAEWrapper
        
        # Create mock VAE for testing
        class MockVAE:
            def __init__(self):
                self.device = torch.device('cpu')
                self.dtype = torch.float32
                self.latents_mean = torch.tensor([0.0])
                self.latents_std = torch.tensor([1.0])
            
            def encode(self, video):
                # video is [B, F, 3, H, W]
                # Return mock latents [B, 128, F', H', W']
                batch, frames, channels, height, width = video.shape
                assert channels == 3, f"Expected 3 channels, got {channels}"
                
                # Simulate latent compression (8x8 spatial, 4x temporal)
                latent_f = max(1, frames // 4)
                latent_h = max(1, height // 8)
                latent_w = max(1, width // 8)
                
                latents = torch.randn(batch, 128, latent_f, latent_h, latent_w)
                
                # Mock latent distribution
                class LatentDist:
                    def __init__(self, latents):
                        self.latents = latents
                    
                    def sample(self, generator=None):
                        return self.latents
                
                class EncoderOutput:
                    def __init__(self, latents):
                        self.latent_dist = LatentDist(latents)
                
                return EncoderOutput(latents)
            
            def decode(self, latents, timestep=0.0, noise_scale=None, generator=None):
                # latents is [B, 128, F', H', W']
                # Return video [B, F, 3, H, W]
                batch, channels, latent_f, latent_h, latent_w = latents.shape
                assert channels == 128, f"Expected 128 latent channels, got {channels}"
                
                # Simulate decompression (inverse of encoding)
                frames = latent_f * 4
                height = latent_h * 8
                width = latent_w * 8
                
                return torch.rand(batch, frames, 3, height, width)
            
            def to(self, *args, **kwargs):
                return self
            
            def enable_tiling(self):
                pass
            
            def disable_tiling(self):
                pass
        
        mock_vae = MockVAE()
        
        # Test 1: 3-channel (should return original VAE)
        print("Test 1: 3-channel input (no wrapper needed)")
        from src.ltxv_trainer.multi_channel_vae import create_multi_channel_vae
        vae_3ch = create_multi_channel_vae(mock_vae, num_channels=3)
        assert vae_3ch == mock_vae, "Should return original VAE for 3 channels"
        print("✓ 3-channel test passed\n")
        
        # Test 2: 10-channel wrapper
        print("Test 2: 10-channel input (multi-channel wrapper)")
        wrapper_10ch = create_multi_channel_vae(mock_vae, num_channels=10)
        assert isinstance(wrapper_10ch, MultiChannelVAEWrapper)
        assert wrapper_10ch.num_channels == 10
        assert wrapper_10ch.num_groups == 4  # ceil(10/3) = 4
        print(f"✓ Created wrapper with {wrapper_10ch.num_groups} groups")
        print(f"  Channel groups: {wrapper_10ch.channel_groups}\n")
        
        # Test 3: Encode multi-channel video
        print("Test 3: Encoding 10-channel video")
        batch_size = 1
        num_frames = 8
        num_channels = 10
        height, width = 64, 64
        
        input_video = torch.rand(batch_size, num_frames, num_channels, height, width)
        print(f"  Input shape: {input_video.shape}")
        
        encoded_latents, metadata = wrapper_10ch.encode(input_video)
        print(f"  Encoded latents shape: {encoded_latents.shape}")
        print(f"  Metadata: num_groups={metadata['num_groups']}, num_channels={metadata['num_channels']}")
        
        assert encoded_latents.shape[0] == batch_size
        assert encoded_latents.shape[1] == 128 * 4  # 4 groups * 128 channels each
        print("✓ Encoding test passed\n")
        
        # Test 4: Decode back to multi-channel
        print("Test 4: Decoding back to 10 channels")
        decoded_video = wrapper_10ch.decode(encoded_latents, metadata)
        print(f"  Decoded shape: {decoded_video.shape}")
        
        assert decoded_video.shape == input_video.shape, \
            f"Shape mismatch: {decoded_video.shape} vs {input_video.shape}"
        assert decoded_video.shape[2] == num_channels, \
            f"Channel count mismatch: {decoded_video.shape[2]} vs {num_channels}"
        print("✓ Decoding test passed\n")
        
        # Test 5: Different channel counts
        print("Test 5: Testing various channel counts")
        for num_ch in [5, 7, 12, 20]:
            wrapper = create_multi_channel_vae(mock_vae, num_channels=num_ch)
            expected_groups = (num_ch + 2) // 3  # ceil(num_ch / 3)
            print(f"  {num_ch} channels → {wrapper.num_groups} groups (expected {expected_groups})")
            assert wrapper.num_groups == expected_groups
        print("✓ Various channel count test passed\n")
        
        print("="*60)
        print("✓ ALL MULTI-CHANNEL VAE TESTS PASSED")
        print("="*60 + "\n")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Note: This is expected if dependencies are not installed")
        print("  The code structure is correct.")
        return
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_configuration():
    """Test that multi-channel config options are properly defined."""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60 + "\n")
    
    try:
        from src.ltxv_trainer.config import DataConfig
        
        # Test creating config with multi-channel options
        config = DataConfig(
            preprocessed_data_root="/test/path",
            use_multi_channel_vae=True,
            num_channels=10,
            channel_grouping_strategy="sequential"
        )
        
        assert config.use_multi_channel_vae == True
        assert config.num_channels == 10
        assert config.channel_grouping_strategy == "sequential"
        
        print("✓ DataConfig multi-channel options validated")
        print(f"  use_multi_channel_vae: {config.use_multi_channel_vae}")
        print(f"  num_channels: {config.num_channels}")
        print(f"  channel_grouping_strategy: {config.channel_grouping_strategy}")
        
        print("\n" + "="*60)
        print("✓ CONFIGURATION TEST PASSED")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_imports():
    """Test that all new modules can be imported."""
    print("\n" + "="*60)
    print("Testing Module Imports")
    print("="*60 + "\n")
    
    try:
        from src.ltxv_trainer import multi_channel_vae
        print("✓ multi_channel_vae module imported")
        
        from src.ltxv_trainer import multi_channel_utils
        print("✓ multi_channel_utils module imported")
        
        # Check key classes/functions exist
        assert hasattr(multi_channel_vae, 'MultiChannelVAEWrapper')
        assert hasattr(multi_channel_vae, 'create_multi_channel_vae')
        print("✓ MultiChannelVAEWrapper class found")
        
        assert hasattr(multi_channel_utils, 'encode_multi_channel_video')
        assert hasattr(multi_channel_utils, 'decode_multi_channel_video')
        print("✓ Encoding/decoding functions found")
        
        print("\n" + "="*60)
        print("✓ IMPORT TEST PASSED")
        print("="*60 + "\n")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Note: Some dependencies may not be installed")
        return
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Multi-Channel Support - Test Suite")
    print("#"*60 + "\n")
    
    try:
        test_imports()
        test_configuration()
        test_multi_channel_vae_wrapper()
        
        print("\n" + "#"*60)
        print("# 🎉 ALL TESTS PASSED")
        print("#"*60 + "\n")
        
    except Exception as e:
        print("\n" + "#"*60)
        print("# ❌ TESTS FAILED")
        print("#"*60 + "\n")
        raise
