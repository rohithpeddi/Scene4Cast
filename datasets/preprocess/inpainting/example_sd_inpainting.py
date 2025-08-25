#!/usr/bin/env python3
"""
Example usage of Stable Diffusion-based video inpainting

This script demonstrates how to use the new VideoInpainterSD class
for high-quality video inpainting using ProPainter and DiffuEraser.
"""

from video_diffuser import VideoInpainterSD
import os


def main():
    # Example configuration
    input_video = "examples/input_video.mp4"
    input_mask = "examples/input_mask.mp4"
    save_dir = "results/sd_inpainting_output"
    video_name = "sd_enhanced_video"

    # Model paths (adjust these based on your setup)
    base_model_path = "weights/stable-diffusion-v1-5"
    vae_path = "weights/sd-vae-ft-mse"
    diffueraser_path = "weights/diffuEraser"
    propainter_model_dir = "weights/propainter"

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
        # Create VideoInpainterSD instance
        print("Initializing VideoInpainterSD with Stable Diffusion models...")
        inpainter = VideoInpainterSD(
            input_video_path=input_video,
            input_mask_path=input_mask,
            save_dir_path=save_dir,
            video_name=video_name,
            base_model_path=base_model_path,
            vae_path=vae_path,
            diffueraser_path=diffueraser_path,
            propainter_model_dir=propainter_model_dir
        )

        # Example 1: Basic inpainting with default parameters
        print("\n" + "=" * 60)
        print("EXAMPLE 1: Basic Inpainting with Default Parameters")
        print("=" * 60)

        results = inpainter.run_inpainting()

        print("\nBasic inpainting completed!")
        print(f"Priori video: {results['priori_path']}")
        print(f"Final video: {results['final_path']}")
        print(f"ProPainter time: {results['priori_time']:.2f}s")
        print(f"DiffuEraser time: {results['diffu_time']:.2f}s")

        # Example 2: High-quality inpainting with custom parameters
        print("\n" + "=" * 60)
        print("EXAMPLE 2: High-Quality Inpainting with Custom Parameters")
        print("=" * 60)

        high_quality_results = inpainter.run_inpainting(
            video_length=None,  # Process full video
            max_img_size=1280,  # Higher resolution for better quality
            mask_dilation_iter=12,  # More dilation for smoother edges
            ref_stride=8,  # Smaller stride for more reference frames
            neighbor_length=15,  # More neighboring frames
            subvideo_length=40  # Shorter sub-videos for better memory management
        )

        print("\nHigh-quality inpainting completed!")
        print(f"Priori video: {high_quality_results['priori_path']}")
        print(f"Final video: {high_quality_results['final_path']}")
        print(f"ProPainter time: {high_quality_results['priori_time']:.2f}s")
        print(f"DiffuEraser time: {high_quality_results['diffu_time']:.2f}s")

        # Example 3: Fast inpainting for quick results
        print("\n" + "=" * 60)
        print("EXAMPLE 3: Fast Inpainting for Quick Results")
        print("=" * 60)

        fast_results = inpainter.run_inpainting(
            video_length=50,  # Process only first 50 frames
            max_img_size=640,  # Lower resolution for speed
            mask_dilation_iter=4,  # Fewer dilation iterations
            ref_stride=15,  # Larger stride for fewer reference frames
            neighbor_length=5,  # Fewer neighboring frames
            subvideo_length=25  # Shorter sub-videos
        )

        print("\nFast inpainting completed!")
        print(f"Priori video: {fast_results['priori_path']}")
        print(f"Final video: {fast_results['final_path']}")
        print(f"ProPainter time: {fast_results['priori_time']:.2f}s")
        print(f"DiffuEraser time: {fast_results['diffu_time']:.2f}s")

        # Example 4: Memory-efficient inpainting for long videos
        print("\n" + "=" * 60)
        print("EXAMPLE 4: Memory-Efficient Inpainting for Long Videos")
        print("=" * 60)

        memory_efficient_results = inpainter.run_inpainting(
            video_length=None,  # Process full video
            max_img_size=720,  # Moderate resolution
            mask_dilation_iter=6,  # Moderate dilation
            ref_stride=20,  # Larger stride to reduce memory usage
            neighbor_length=8,  # Moderate neighbor count
            subvideo_length=30  # Short sub-videos for memory efficiency
        )

        print("\nMemory-efficient inpainting completed!")
        print(f"Priori video: {memory_efficient_results['priori_path']}")
        print(f"Final video: {memory_efficient_results['final_path']}")
        print(f"ProPainter time: {memory_efficient_results['priori_time']:.2f}s")
        print(f"DiffuEraser time: {memory_efficient_results['diffu_time']:.2f}s")

        # Example 5: Video analysis and information
        print("\n" + "=" * 60)
        print("EXAMPLE 5: Video Analysis and Information")
        print("=" * 60)

        print(f"Video Properties:")
        print(f"  Frame count: {inpainter.frame_count}")
        print(f"  Frame size: {inpainter.frame_size}")
        print(f"  FPS: {inpainter.fps:.2f}")
        print(f"  Duration: {inpainter.frame_count / inpainter.fps:.2f} seconds")

        print(f"\nModel Information:")
        print(f"  Base model: {os.path.basename(base_model_path)}")
        print(f"  VAE model: {os.path.basename(vae_path)}")
        print(f"  DiffuEraser: {os.path.basename(diffueraser_path)}")
        print(f"  ProPainter: {os.path.basename(propainter_model_dir)}")

        # Example 6: Performance comparison
        print("\n" + "=" * 60)
        print("EXAMPLE 6: Performance Comparison")
        print("=" * 60)

        print("Performance comparison across different configurations:")
        print(f"{'Configuration':<25} {'Total Time':<12} {'Quality':<10}")
        print("-" * 50)

        configs = [
            ("Fast", fast_results, "Low"),
            ("Memory Efficient", memory_efficient_results, "Medium"),
            ("Default", results, "Medium"),
            ("High Quality", high_quality_results, "High")
        ]

        for name, result, quality in configs:
            total_time = result['priori_time'] + result['diffu_time']
            print(f"{name:<25} {total_time:<12.2f}s {quality:<10}")

    except FileNotFoundError as e:
        print(f"Model not found: {e}")
        print("Please run 'python download_models.py' to download required models")
    except Exception as e:
        print(f"Error during inpainting: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_sd_capabilities():
    """Demonstrate the Stable Diffusion inpainting capabilities."""
    print("\n" + "=" * 60)
    print("STABLE DIFFUSION INPAINTING CAPABILITIES")
    print("=" * 60)

    print("\n1. ProPainter Stage (Temporal Consistency):")
    print("   - Flow-based prior generation for temporal coherence")
    print("   - Handles complex motion and camera movement")
    print("   - Configurable reference frame selection")

    print("\n2. DiffuEraser Stage (Quality Refinement):")
    print("   - Stable Diffusion-based refinement")
    print("   - High-quality visual results")
    print("   - Configurable guidance and sampling parameters")

    print("\n3. Two-Stage Pipeline:")
    print("   - Stage 1: ProPainter generates temporally consistent priori")
    print("   - Stage 2: DiffuEraser refines for visual quality")
    print("   - Optimal balance of speed and quality")

    print("\n4. Configurable Parameters:")
    print("   - Video length and resolution control")
    print("   - Mask dilation and processing options")
    print("   - Reference frame and neighbor selection")
    print("   - Sub-video length for memory management")


def show_sd_parameters():
    """Show available Stable Diffusion inpainting parameters."""
    print("\n" + "=" * 60)
    print("STABLE DIFFUSION INPAINTING PARAMETERS")
    print("=" * 60)

    print("\nVideo Processing Parameters:")
    print("  video_length: Maximum frames to process (None = full video)")
    print("  max_img_size: Maximum frame dimension for processing")

    print("\nMask Processing Parameters:")
    print("  mask_dilation_iter: Number of mask dilation iterations")

    print("\nProPainter Parameters:")
    print("  ref_stride: Temporal stride for reference frames")
    print("  neighbor_length: Number of neighboring frames")
    print("  subvideo_length: Length of sub-videos")

    print("\nQuality vs Speed Trade-offs:")
    print("  Higher resolution → Better quality, slower processing")
    print("  More dilation → Smoother edges, more processing time")
    print("  Smaller stride → More reference frames, better consistency")
    print("  More neighbors → Better temporal coherence, more memory")


def show_optimization_tips():
    """Show optimization tips for different use cases."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION TIPS")
    print("=" * 60)

    print("\nFor High Quality:")
    print("  - Use max_img_size >= 1280")
    print("  - Set mask_dilation_iter >= 10")
    print("  - Use ref_stride <= 8")
    print("  - Set neighbor_length >= 12")

    print("\nFor Fast Processing:")
    print("  - Use max_img_size <= 640")
    print("  - Set mask_dilation_iter <= 4")
    print("  - Use ref_stride >= 15")
    print("  - Set neighbor_length <= 6")

    print("\nFor Memory Efficiency:")
    print("  - Use subvideo_length <= 30")
    print("  - Limit max_img_size to 720")
    print("  - Use larger ref_stride values")
    print("  - Process videos in segments")


if __name__ == "__main__":
    print("Stable Diffusion Video Inpainting Examples")
    print("=" * 60)

    # Demonstrate capabilities first
    demonstrate_sd_capabilities()

    # Show available parameters
    show_sd_parameters()

    # Show optimization tips
    show_optimization_tips()

    # Run examples
    main()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nKey benefits of Stable Diffusion inpainting:")
    print("✓ High-quality visual results using diffusion models")
    print("✓ Temporal consistency through ProPainter")
    print("✓ Configurable parameters for different use cases")
    print("✓ Memory-efficient processing for long videos")
    print("✓ Professional-grade inpainting quality")
    print("\nFor more information, see the VideoInpainterSD class documentation.")
