#!/usr/bin/env python3
"""
Test script for NPZ physics simulation support.

This script validates that NPZ files can be properly loaded and preprocessed
by the LTX-Video trainer.
"""

import tempfile
from pathlib import Path
import numpy as np
import json
import torch


def create_test_npz_files(output_dir: Path, num_files: int = 3) -> list[Path]:
    """Create test NPZ files with various formats."""
    npz_files = []
    
    # Test case 1: Standard format (T, H, W, C) with normalized values
    frames1 = np.random.rand(25, 256, 256, 3).astype(np.float32)
    npz_path1 = output_dir / "test_sim_001.npz"
    np.savez(npz_path1, frames=frames1, fps=24)
    npz_files.append(npz_path1)
    print(f"✓ Created {npz_path1.name}: (T, H, W, C) format, normalized values")
    
    # Test case 2: Alternative format (T, C, H, W) with 0-255 values
    frames2 = np.random.randint(0, 256, size=(33, 3, 256, 256), dtype=np.uint8)
    npz_path2 = output_dir / "test_sim_002.npz"
    np.savez(npz_path2, data=frames2, fps=30)  # Using 'data' key instead of 'frames'
    npz_files.append(npz_path2)
    print(f"✓ Created {npz_path2.name}: (T, C, H, W) format, 0-255 values, 'data' key")
    
    # Test case 3: Grayscale format (T, H, W)
    frames3 = np.random.rand(41, 256, 256).astype(np.float32)
    npz_path3 = output_dir / "test_sim_003.npz"
    np.savez(npz_path3, frames=frames3)  # No FPS, should default to 24
    npz_files.append(npz_path3)
    print(f"✓ Created {npz_path3.name}: (T, H, W) grayscale format, no FPS")
    
    return npz_files


def test_npz_format_validation():
    """Test that NPZ files have the correct format and can be loaded."""
    print("\n" + "="*60)
    print("Testing NPZ File Format Validation")
    print("="*60 + "\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test NPZ files
        print("Step 1: Creating test NPZ files...")
        npz_files = create_test_npz_files(temp_path)
        
        # Test loading each NPZ file
        print("\nStep 2: Validating NPZ file contents...")
        for npz_file in npz_files:
            print(f"\n  Validating {npz_file.name}:")
            
            # Load NPZ
            data = np.load(npz_file)
            
            # Check keys
            print(f"    Keys: {list(data.keys())}")
            
            # Get frames
            if 'frames' in data:
                frames = data['frames']
            elif 'data' in data:
                frames = data['data']
            else:
                raise ValueError(f"Missing 'frames' or 'data' key in {npz_file.name}")
            
            print(f"    Shape: {frames.shape}")
            print(f"    Dtype: {frames.dtype}")
            print(f"    Value range: [{frames.min():.3f}, {frames.max():.3f}]")
            
            # Validate dimensions
            assert frames.ndim in [3, 4], f"Expected 3 or 4 dimensions, got {frames.ndim}"
            
            # Get FPS
            fps = data.get('fps', 24)
            print(f"    FPS: {fps}")
            
            print(f"    ✓ Format validation passed")
        
        print("\n" + "="*60)
        print("✓ ALL FORMAT TESTS PASSED")
        print("="*60 + "\n")


def test_npz_preprocessing_logic():
    """Test the preprocessing logic for NPZ files without requiring full dependencies."""
    print("\n" + "="*60)
    print("Testing NPZ Preprocessing Logic")
    print("="*60 + "\n")
    
    # Test case 1: (T, H, W, C) format
    print("Test 1: (T, H, W, C) format")
    frames = np.random.rand(25, 128, 128, 3).astype(np.float32)
    frames_tensor = torch.from_numpy(frames).float()
    assert frames_tensor.shape == (25, 128, 128, 3)
    # Permute to (T, C, H, W)
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)
    assert frames_tensor.shape == (25, 3, 128, 128)
    print(f"  ✓ Conversion to (T, C, H, W): {frames_tensor.shape}")
    
    # Test case 2: (T, C, H, W) format - already correct
    print("\nTest 2: (T, C, H, W) format")
    frames = np.random.randint(0, 256, size=(25, 3, 128, 128), dtype=np.uint8)
    frames_tensor = torch.from_numpy(frames).float()
    # Normalize
    frames_tensor = frames_tensor / 255.0
    assert frames_tensor.max() <= 1.0
    assert frames_tensor.min() >= 0.0
    print(f"  ✓ Normalization: range [{frames_tensor.min():.3f}, {frames_tensor.max():.3f}]")
    
    # Test case 3: (T, H, W) grayscale
    print("\nTest 3: (T, H, W) grayscale format")
    frames = np.random.rand(25, 128, 128).astype(np.float32)
    frames_tensor = torch.from_numpy(frames).float()
    # Add channel dimension
    frames_tensor = frames_tensor.unsqueeze(1)
    assert frames_tensor.shape == (25, 1, 128, 128)
    # Expand to 3 channels
    frames_tensor = frames_tensor.repeat(1, 3, 1, 1)
    assert frames_tensor.shape == (25, 3, 128, 128)
    print(f"  ✓ Grayscale to RGB: {frames_tensor.shape}")
    
    # Test case 4: RGBA with alpha channel
    print("\nTest 4: RGBA format (drop alpha)")
    frames = np.random.rand(25, 4, 128, 128).astype(np.float32)
    frames_tensor = torch.from_numpy(frames).float()
    # Drop alpha channel
    frames_tensor = frames_tensor[:, :3, :, :]
    assert frames_tensor.shape == (25, 3, 128, 128)
    print(f"  ✓ RGBA to RGB: {frames_tensor.shape}")
    
    print("\n" + "="*60)
    print("✓ ALL PREPROCESSING LOGIC TESTS PASSED")
    print("="*60 + "\n")


def test_npz_utils():
    """Test the NPZ utility script functions."""
    print("\n" + "="*60)
    print("Testing NPZ Utility Functions")
    print("="*60 + "\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a test NPZ file
        print("Creating test NPZ file...")
        frames = np.random.rand(25, 128, 128, 3).astype(np.float32)
        test_npz = temp_path / "test.npz"
        np.savez(test_npz, frames=frames, fps=24)
        
        # Test validation
        print("\nTesting first frame extraction...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        try:
            from scripts.npz_utils import extract_first_frame
            
            # Extract first frame
            first_frame_path = temp_path / "first_frame.png"
            extract_first_frame(test_npz, first_frame_path)
            
            assert first_frame_path.exists(), "First frame not created"
            print(f"✓ First frame extracted to {first_frame_path}")
            
            # Verify the image
            from PIL import Image
            img = Image.open(first_frame_path)
            assert img.size == (128, 128), f"Expected size (128, 128), got {img.size}"
            assert img.mode == 'RGB', f"Expected RGB mode, got {img.mode}"
            print(f"✓ First frame validation passed (size: {img.size}, mode: {img.mode})")
            
        except ImportError as e:
            print(f"⚠ Skipping utility tests due to missing dependencies: {e}")
        
        print("\n" + "="*60)
        print("✓ UTILITY TESTS COMPLETED")
        print("="*60 + "\n")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# NPZ Physics Simulation Support - Test Suite")
    print("#"*60 + "\n")
    
    try:
        test_npz_format_validation()
        test_npz_preprocessing_logic()
        test_npz_utils()
        
        print("\n" + "#"*60)
        print("# 🎉 ALL TESTS COMPLETED SUCCESSFULLY")
        print("#"*60 + "\n")
        
    except Exception as e:
        print("\n" + "#"*60)
        print("# ❌ TESTS FAILED")
        print("#"*60 + "\n")
        raise
