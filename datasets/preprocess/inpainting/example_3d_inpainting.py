#!/usr/bin/env python3
"""
Example usage of 3D-aware video inpainting

This script demonstrates how to use the new VideoInpainter3D class
for 3D scene representation and geometric consistency in monocular videos.
"""

from video_diffuser import VideoInpainter3D
import os
import numpy as np


def main():
    # Example configuration
    input_video = "examples/input_video.mp4"
    input_mask = "examples/input_mask.mp4"
    save_dir = "results/3d_inpainting_output"
    video_name = "3d_enhanced_video"

    # Check if input files exist
    if not os.path.exists(input_video):
        print(f"Input video not found: {input_video}")
        print("Please provide a valid input video path")
        return

    if not os.path.exists(input_mask):
        print(f"Input mask not found: {input_mask}")
        print("Please provide a valid input mask path")
        return

    try:
        # Create VideoInpainter3D instance with custom 3D parameters
        print("Initializing VideoInpainter3D with 3D capabilities...")
        inpainter = VideoInpainter3D(
            input_video_path=input_video,
            input_mask_path=input_mask,
            save_dir_path=save_dir,
            video_name=video_name,
            depth_estimation_model='monodepth',
            flow_estimation_model='raft',
            min_parallax_threshold=5.0,
            max_depth_discontinuity=0.1
        )

        # Example 1: Basic 3D-aware inpainting
        print("\n" + "=" * 60)
        print("EXAMPLE 1: Basic 3D-Aware Inpainting")
        print("=" * 60)

        results = inpainter.run_3d_aware_inpainting()

        print("\n3D-aware inpainting completed!")
        print(f"Priori video: {results['priori_path']}")
        print(f"Final video: {results['final_path']}")
        print(f"Priori generation time: {results['priori_time']:.2f}s")
        print(f"Refinement time: {results['refinement_time']:.2f}s")

        # Example 2: Access 3D scene information
        print("\n" + "=" * 60)
        print("EXAMPLE 2: 3D Scene Information")
        print("=" * 60)

        print(f"3D Scene Model Type: {inpainter.static_scene_model['type']}")
        print(f"Depth Maps: {len(inpainter.depth_maps)} frames")
        print(f"Optical Flow: {len(inpainter.optical_flow)} frame pairs")
        print(f"Camera Parameters: {len(inpainter.camera_params['extrinsics'])} poses")

        # Example 3: Analyze depth information
        print("\n" + "=" * 60)
        print("EXAMPLE 3: Depth Analysis")
        print("=" * 60)

        if inpainter.depth_maps:
            # Analyze depth statistics for first frame
            first_depth = inpainter.depth_maps[0]
            print(f"First frame depth statistics:")
            print(f"  Min depth: {np.min(first_depth):.3f}")
            print(f"  Max depth: {np.max(first_depth):.3f}")
            print(f"  Mean depth: {np.mean(first_depth):.3f}")
            print(f"  Depth range: {np.max(first_depth) - np.min(first_depth):.3f}")

        # Example 4: Camera motion analysis
        print("\n" + "=" * 60)
        print("EXAMPLE 4: Camera Motion Analysis")
        print("=" * 60)

        if inpainter.camera_params and inpainter.camera_params['extrinsics']:
            print("Camera motion analysis:")
            for i, pose in enumerate(inpainter.camera_params['extrinsics'][:5]):  # First 5 poses
                translation = pose['translation']
                confidence = pose['confidence']
                print(f"  Frame {i}: Translation {translation}, Confidence: {confidence:.2f}")

        # Example 5: Optical flow analysis
        print("\n" + "=" * 60)
        print("EXAMPLE 5: Optical Flow Analysis")
        print("=" * 60)

        if inpainter.optical_flow:
            print("Optical flow analysis:")
            for i, flow in enumerate(inpainter.optical_flow[:3]):  # First 3 flows
                flow_magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
                avg_magnitude = np.mean(flow_magnitude)
                max_magnitude = np.max(flow_magnitude)
                print(f"  Flow {i + 1}: Avg magnitude: {avg_magnitude:.2f}, Max: {max_magnitude:.2f}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check your input file paths")
    except Exception as e:
        print(f"Error during 3D inpainting: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_3d_capabilities():
    """Demonstrate the 3D scene representation capabilities."""
    print("\n" + "=" * 60)
    print("3D SCENE REPRESENTATION CAPABILITIES")
    print("=" * 60)

    print("\n1. Monocular Depth Estimation:")
    print("   - Estimates depth from single camera view")
    print("   - Uses edge detection and image structure analysis")
    print("   - Fallback to simplified depth when advanced models unavailable")

    print("\n2. Optical Flow Estimation:")
    print("   - Tracks pixel motion between consecutive frames")
    print("   - Uses Lucas-Kanade method for feature tracking")
    print("   - Enables temporal consistency and camera motion estimation")

    print("\n3. Camera Parameter Estimation:")
    print("   - Estimates camera intrinsics and extrinsics")
    print("   - Derives motion from optical flow and depth")
    print("   - Enables 3D scene reconstruction")

    print("\n4. 3D Geometric Constraints:")
    print("   - Depth-aware inpainting using similar depth regions")
    print("   - Maintains scene geometry and perspective")
    print("   - Reduces artifacts in complex 3D scenes")

    print("\n5. Temporal Consistency:")
    print("   - Flow-based consistency between frames")
    print("   - Parallax-aware inpainting for camera motion")
    print("   - Smooth transitions across video frames")

    print("\n6. Advanced 3D Processing:")
    print("   - Depth-based smoothing for artifact reduction")
    print("   - Parallax consistency for realistic camera motion")
    print("   - Scene-aware interpolation and blending")


def show_3d_parameters():
    """Show available 3D processing parameters."""
    print("\n" + "=" * 60)
    print("3D PROCESSING PARAMETERS")
    print("=" * 60)

    print("\nDepth Estimation Parameters:")
    print("  depth_estimation_model: 'monodepth' | 'simplified'")
    print("  max_depth_discontinuity: 0.1 (threshold for depth consistency)")

    print("\nOptical Flow Parameters:")
    print("  flow_estimation_model: 'raft' | 'lucas_kanade'")
    print("  min_parallax_threshold: 5.0 (minimum motion for 3D reconstruction)")

    print("\n3D Scene Parameters:")
    print("  search_radius: 50 (pixels for finding similar depth regions)")
    print("  max_similar_pixels: 20 (candidates for depth-based inpainting)")
    print("  temporal_consistency_threshold: 50 (pixel difference threshold)")

    print("\nSmoothing Parameters:")
    print("  kernel_size: 5 (depth-aware smoothing kernel size)")
    print("  blend_factor: 0.3 (temporal consistency blending strength)")


if __name__ == "__main__":
    print("3D-Aware Video Inpainting Examples")
    print("=" * 60)

    # Demonstrate capabilities first
    demonstrate_3d_capabilities()

    # Show available parameters
    show_3d_parameters()

    # Run examples
    main()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nKey benefits of 3D-aware inpainting:")
    print("✓ Better geometric consistency across camera views")
    print("✓ Improved temporal coherence in dynamic scenes")
    print("✓ Reduced artifacts in videos with camera motion")
    print("✓ Higher quality results for complex 3D scenes")
    print("✓ Monocular depth estimation without external sensors")
    print("✓ Real-time optical flow for temporal consistency")
    print("\nFor more information, see the VideoInpainter3D class documentation.")
