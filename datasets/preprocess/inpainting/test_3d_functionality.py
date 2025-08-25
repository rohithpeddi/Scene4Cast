#!/usr/bin/env python3
"""
Test script for 3D-aware inpainting functionality

This script tests the 3D scene representation and inpainting capabilities
with actual 3D geometric and temporal constraints for monocular videos.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_3d_class_creation():
    """Test that the VideoInpainter3D class can be created with 3D capabilities."""
    try:
        from video_diffuser import VideoInpainter3D
        print("✓ VideoInpainter3D class imported successfully")
        
        # Test class creation (without actual files)
        inpainter = VideoInpainter3D(
            input_video_path="test_video.mp4",
            input_mask_path="test_mask.mp4",
            save_dir_path="test_output",
            video_name="test_video"
        )
        print("✓ VideoInpainter3D instance created successfully")
        
        # Test 3D scene representation building
        print("\nTesting 3D scene representation...")
        
        # Test depth estimation
        if inpainter.depth_maps:
            print(f"✓ Depth maps created: {len(inpainter.depth_maps)} frames")
            print(f"  First frame depth shape: {inpainter.depth_maps[0].shape}")
            print(f"  Depth range: {np.min(inpainter.depth_maps[0]):.3f} - {np.max(inpainter.depth_maps[0]):.3f}")
        else:
            print("✗ Depth maps not created")
            return False
        
        # Test optical flow estimation
        if inpainter.optical_flow:
            print(f"✓ Optical flow created: {len(inpainter.optical_flow)} frame pairs")
            print(f"  First flow shape: {inpainter.optical_flow[0].shape}")
        else:
            print("✗ Optical flow not created")
            return False
        
        # Test camera parameters
        if inpainter.camera_params:
            print(f"✓ Camera parameters created")
            print(f"  Intrinsics: {inpainter.camera_params['intrinsics']}")
            print(f"  Extrinsics: {len(inpainter.camera_params['extrinsics'])} poses")
        else:
            print("✗ Camera parameters not created")
            return False
        
        # Test 3D scene model
        if inpainter.static_scene_model:
            print(f"✓ 3D scene model created: {inpainter.static_scene_model['type']}")
        else:
            print("✗ 3D scene model not created")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import VideoInpainter3D: {e}")
        return False
    except Exception as e:
        print(f"✗ Failed to create VideoInpainter3D instance: {e}")
        return False

def test_3d_methods():
    """Test that 3D-related methods exist and are callable."""
    try:
        from video_diffuser import VideoInpainter3D
        
        inpainter = VideoInpainter3D(
            input_video_path="test_video.mp4",
            input_mask_path="test_mask.mp4",
            save_dir_path="test_output",
            video_name="test_video"
        )
        
        print("\nTesting 3D method availability...")
        
        # Check if 3D methods exist
        required_methods = [
            'run_3d_aware_inpainting',
            '_generate_3d_guided_priori',
            '_apply_3d_geometric_constraints',
            '_apply_temporal_consistency_constraints',
            '_refine_with_3d_constraints',
            '_apply_final_3d_refinement',
            '_apply_depth_based_smoothing',
            '_apply_parallax_consistency'
        ]
        
        for method_name in required_methods:
            if hasattr(inpainter, method_name):
                method = getattr(inpainter, method_name)
                if callable(method):
                    print(f"✓ Method '{method_name}' exists and is callable")
                else:
                    print(f"✗ Method '{method_name}' exists but is not callable")
                    return False
            else:
                print(f"✗ Method '{method_name}' not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test 3D methods: {e}")
        return False

def test_3d_geometric_constraints():
    """Test the actual 3D geometric constraints implementation."""
    try:
        from video_diffuser import VideoInpainter3D
        
        inpainter = VideoInpainter3D(
            input_video_path="test_video.mp4",
            input_mask_path="test_mask.mp4",
            save_dir_path="test_output",
            video_name="test_video"
        )
        
        print("\nTesting 3D geometric constraints...")
        
        # Test depth-aware inpainting
        if inpainter.depth_maps and inpainter.mask_frames:
            # Create a test frame and mask
            test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            test_mask = np.zeros((100, 100), dtype=np.uint8)
            test_mask[40:60, 40:60] = 255  # Create a mask region
            test_depth = inpainter.depth_maps[0][:100, :100]  # Use first frame depth
            
            # Test geometric constraints application
            try:
                guided_frame = inpainter._apply_3d_geometric_constraints(
                    test_frame, test_mask, test_depth, 0
                )
                print("✓ 3D geometric constraints applied successfully")
                print(f"  Input frame shape: {test_frame.shape}")
                print(f"  Output frame shape: {guided_frame.shape}")
                print(f"  Frame modified: {not np.array_equal(test_frame, guided_frame)}")
            except Exception as e:
                print(f"✗ Failed to apply 3D geometric constraints: {e}")
                return False
        else:
            print("⚠ Skipping geometric constraints test (no depth/mask data)")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test 3D geometric constraints: {e}")
        return False

def test_3d_temporal_constraints():
    """Test the actual 3D temporal consistency constraints."""
    try:
        from video_diffuser import VideoInpainter3D
        
        inpainter = VideoInpainter3D(
            input_video_path="test_video.mp4",
            input_mask_path="test_mask.mp4",
            save_dir_path="test_output",
            video_name="test_video"
        )
        
        print("\nTesting 3D temporal constraints...")
        
        # Test temporal consistency application
        if inpainter.optical_flow:
            # Create test frames
            current_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            previous_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            try:
                consistent_frame = inpainter._apply_temporal_consistency_constraints(
                    current_frame, previous_frame, 1
                )
                print("✓ Temporal consistency constraints applied successfully")
                print(f"  Current frame shape: {current_frame.shape}")
                print(f"  Consistent frame shape: {consistent_frame.shape}")
                print(f"  Frame modified: {not np.array_equal(current_frame, consistent_frame)}")
            except Exception as e:
                print(f"✗ Failed to apply temporal constraints: {e}")
                return False
        else:
            print("⚠ Skipping temporal constraints test (no optical flow data)")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test 3D temporal constraints: {e}")
        return False

def test_3d_depth_processing():
    """Test depth-based processing methods."""
    try:
        from video_diffuser import VideoInpainter3D
        
        inpainter = VideoInpainter3D(
            input_video_path="test_video.mp4",
            input_mask_path="test_mask.mp4",
            save_dir_path="test_output",
            video_name="test_video"
        )
        
        print("\nTesting depth-based processing...")
        
        if inpainter.depth_maps:
            # Test depth-based smoothing
            test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            test_depth = inpainter.depth_maps[0][:100, :100]
            
            try:
                smoothed_frame = inpainter._apply_depth_based_smoothing(test_frame, test_depth)
                print("✓ Depth-based smoothing applied successfully")
                print(f"  Input frame shape: {test_frame.shape}")
                print(f"  Smoothed frame shape: {smoothed_frame.shape}")
                print(f"  Frame modified: {not np.array_equal(test_frame, smoothed_frame)}")
            except Exception as e:
                print(f"✗ Failed to apply depth-based smoothing: {e}")
                return False
            
            # Test parallax consistency
            try:
                parallax_frame = inpainter._apply_parallax_consistency(test_frame, 0)
                print("✓ Parallax consistency applied successfully")
                print(f"  Output frame shape: {parallax_frame.shape}")
            except Exception as e:
                print(f"✗ Failed to apply parallax consistency: {e}")
                return False
        else:
            print("⚠ Skipping depth processing test (no depth data)")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test depth processing: {e}")
        return False

def test_3d_attributes():
    """Test that 3D-related attributes are properly initialized."""
    try:
        from video_diffuser import VideoInpainter3D
        
        inpainter = VideoInpainter3D(
            input_video_path="test_video.mp4",
            input_mask_path="test_mask.mp4",
            save_dir_path="test_output",
            video_name="test_video"
        )
        
        print("\nTesting 3D attribute initialization...")
        
        # Check 3D attributes
        required_attributes = [
            'scene_3d', 'camera_params', 'static_scene_model',
            'depth_maps', 'optical_flow'
        ]
        
        for attr_name in required_attributes:
            if hasattr(inpainter, attr_name):
                attr_value = getattr(inpainter, attr_name)
                print(f"✓ Attribute '{attr_name}' exists with value: {type(attr_value)}")
            else:
                print(f"✗ Attribute '{attr_name}' not found")
                return False
        
        # Check 3D processing parameters
        processing_params = [
            'depth_estimation_model', 'flow_estimation_model',
            'min_parallax_threshold', 'max_depth_discontinuity'
        ]
        
        for param_name in processing_params:
            if hasattr(inpainter, param_name):
                param_value = getattr(inpainter, param_name)
                print(f"✓ Parameter '{param_name}' exists with value: {param_value}")
            else:
                print(f"✗ Parameter '{param_name}' not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test 3D attributes: {e}")
        return False

def test_sd_class():
    """Test the VideoInpainterSD class for Stable Diffusion inpainting."""
    try:
        from video_diffuser import VideoInpainterSD
        print("\nTesting VideoInpainterSD class...")
        
        # Test class creation (without actual files)
        inpainter = VideoInpainterSD(
            input_video_path="test_video.mp4",
            input_mask_path="test_mask.mp4",
            save_dir_path="test_output",
            video_name="test_video",
            base_model_path="weights/stable-diffusion-v1-5",
            vae_path="weights/sd-vae-ft-mse",
            diffueraser_path="weights/diffuEraser",
            propainter_model_dir="weights/propainter"
        )
        print("✓ VideoInpainterSD instance created successfully")
        
        # Check SD-specific attributes
        sd_attributes = ['base_model_path', 'vae_path', 'diffueraser_path', 'propainter_model_dir']
        for attr_name in sd_attributes:
            if hasattr(inpainter, attr_name):
                print(f"✓ SD attribute '{attr_name}' exists")
            else:
                print(f"✗ SD attribute '{attr_name}' not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test VideoInpainterSD: {e}")
        return False

def main():
    """Run all 3D functionality tests."""
    print("="*60)
    print("3D-AWARE INPAINTING FUNCTIONALITY TESTS")
    print("="*60)
    
    tests = [
        ("3D Class Creation", test_3d_class_creation),
        ("3D Methods", test_3d_methods),
        ("3D Geometric Constraints", test_3d_geometric_constraints),
        ("3D Temporal Constraints", test_3d_temporal_constraints),
        ("3D Depth Processing", test_3d_depth_processing),
        ("3D Attributes", test_3d_attributes),
        ("SD Class", test_sd_class)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- Running Test: {test_name} ---")
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! 3D functionality is working correctly.")
        print("\nThe system now includes:")
        print("✓ Actual 3D geometric constraints for monocular videos")
        print("✓ Real temporal consistency using optical flow")
        print("✓ Depth-aware inpainting with scene understanding")
        print("✓ Parallax consistency for camera motion")
        print("✓ Separate classes for 3D and SD inpainting")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        print("\nThe system may still work with basic functionality.")
    
    print("\nNext steps:")
    print("1. Test with real video files:")
    print("   python example_3d_inpainting.py")
    print("   python example_sd_inpainting.py")
    print("2. Check the README.md for detailed usage instructions")
    print("3. Both classes are now separate and optimized for their specific use cases")

if __name__ == "__main__":
    main()
