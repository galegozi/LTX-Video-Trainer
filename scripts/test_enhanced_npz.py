#!/usr/bin/env python3
"""
Test enhanced NPZ features.

This tests the new configuration options and NPZ processing capabilities.
"""

import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_config_loading():
    """Test that enhanced config options load correctly."""
    print("\n" + "="*60)
    print("Testing Configuration Loading")
    print("="*60 + "\n")
    
    from src.ltxv_trainer.config import DataConfig, ValidationConfig
    
    # Test DataConfig with NPZ options
    data_config = DataConfig(
        preprocessed_data_root="/test/path",
        npz_initial_frame_index=10,
        npz_channel_names=["temperature", "pressure", "velocity_x"],
        npz_preserve_all_channels=True
    )
    
    assert data_config.npz_initial_frame_index == 10
    assert data_config.npz_channel_names == ["temperature", "pressure", "velocity_x"]
    assert data_config.npz_preserve_all_channels == True
    print("✓ DataConfig NPZ options loaded correctly")
    
    # Test ValidationConfig with new options
    val_config = ValidationConfig(
        prompts=["test"],
        output_fps=20.0,
        visualization_script="/path/to/script.py",
        save_comparison_gif=True
    )
    
    assert val_config.output_fps == 20.0
    assert val_config.visualization_script == "/path/to/script.py"
    assert val_config.save_comparison_gif == True
    print("✓ ValidationConfig enhanced options loaded correctly")
    
    print("\n" + "="*60)
    print("✓ CONFIGURATION TESTS PASSED")
    print("="*60 + "\n")


def test_npz_visualization_imports():
    """Test that NPZ visualization module can be imported."""
    print("\n" + "="*60)
    print("Testing NPZ Visualization Module")
    print("="*60 + "\n")
    
    try:
        from src.ltxv_trainer import npz_visualization
        
        # Check that functions exist
        assert hasattr(npz_visualization, 'visualize_density_sum')
        assert hasattr(npz_visualization, 'create_comparison_gif')
        assert hasattr(npz_visualization, 'run_custom_visualization_script')
        assert hasattr(npz_visualization, 'load_npz_sequence')
        
        print("✓ NPZ visualization module imports successfully")
        print("✓ All required functions available")
        
        print("\n" + "="*60)
        print("✓ NPZ VISUALIZATION TESTS PASSED")
        print("="*60 + "\n")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        raise


def test_media_dataset_init():
    """Test that MediaDataset accepts new parameters."""
    print("\n" + "="*60)
    print("Testing MediaDataset Initialization")
    print("="*60 + "\n")
    
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create dummy dataset file
        dataset_file = tmppath / "dataset.json"
        with open(dataset_file, 'w') as f:
            json.dump([], f)
        
        try:
            # Import after creating test file
            sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
            from process_videos import MediaDataset
            
            # Test initialization with new parameters
            dataset = MediaDataset(
                dataset_file=dataset_file,
                main_media_column="media_path",
                video_column="media_path",
                resolution_buckets=[(25, 256, 256)],
                reshape_mode="center",
                initial_frame_index=10,  # NEW
                preserve_all_channels=True  # NEW
            )
            
            assert dataset.initial_frame_index == 10
            assert dataset.preserve_all_channels == True
            
            print("✓ MediaDataset accepts initial_frame_index parameter")
            print("✓ MediaDataset accepts preserve_all_channels parameter")
            
            print("\n" + "="*60)
            print("✓ MEDIADATASET TESTS PASSED")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            raise


def test_config_file_validation():
    """Test that enhanced config file can be loaded."""
    print("\n" + "="*60)
    print("Testing Enhanced Config File")
    print("="*60 + "\n")
    
    config_path = Path(__file__).parent.parent / "configs" / "physics_simulation_enhanced.yaml"
    
    if not config_path.exists():
        print(f"⚠ Config file not found: {config_path}")
        return
    
    try:
        import yaml
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check key sections exist
        assert 'data' in config
        assert 'validation' in config
        
        # Check NPZ-specific data settings
        if 'data' in config:
            data = config['data']
            print(f"  npz_initial_frame_index: {data.get('npz_initial_frame_index', 'not set')}")
            print(f"  npz_preserve_all_channels: {data.get('npz_preserve_all_channels', 'not set')}")
        
        # Check validation settings
        if 'validation' in config:
            val = config['validation']
            print(f"  output_fps: {val.get('output_fps', 'not set')}")
            print(f"  save_comparison_gif: {val.get('save_comparison_gif', 'not set')}")
        
        print("\n✓ Enhanced config file is valid YAML")
        print("✓ All required sections present")
        
        print("\n" + "="*60)
        print("✓ CONFIG FILE TESTS PASSED")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        raise


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Enhanced NPZ Features - Test Suite")
    print("#"*60 + "\n")
    
    try:
        test_config_loading()
        test_npz_visualization_imports()
        test_config_file_validation()
        test_media_dataset_init()
        
        print("\n" + "#"*60)
        print("# 🎉 ALL TESTS PASSED")
        print("#"*60 + "\n")
        
    except Exception as e:
        print("\n" + "#"*60)
        print("# ❌ TESTS FAILED")
        print("#"*60 + "\n")
        raise
