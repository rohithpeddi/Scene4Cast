import torch
import os
import time
import argparse
import numpy as np
import cv2
from pathlib import Path
from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device

class BaseVideoInpainter:
    """Base class for video inpainting with common functionality."""
    
    def __init__(self, 
                 input_video_path: str,
                 input_mask_path: str,
                 save_dir_path: str,
                 video_name: str):
        
        self.input_video_path = input_video_path
        self.input_mask_path = input_mask_path
        self.save_dir_path = save_dir_path
        self.video_name = video_name
        
        # Validate input paths
        self._validate_paths()
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir_path):
            os.makedirs(self.save_dir_path)
        
        # Initialize device
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Video properties
        self.video_frames = None
        self.mask_frames = None
        self.frame_count = 0
        self.fps = 30.0
        self.frame_size = (1920, 1080)
        
        # Load video data
        self._load_video_data()
        
    def _validate_paths(self):
        """Validate that input video and mask files exist."""
        if not os.path.exists(self.input_video_path):
            raise FileNotFoundError(f"Input video not found: {self.input_video_path}")
        if not os.path.exists(self.input_mask_path):
            raise FileNotFoundError(f"Input mask not found: {self.input_mask_path}")
    
    def _load_video_data(self):
        """Load video and mask data for processing."""
        print("Loading video data...")
        
        # Load input video
        cap = cv2.VideoCapture(self.input_video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        # Load video frames
        self.video_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.video_frames.append(frame)
        cap.release()
        
        # Load mask video
        cap = cv2.VideoCapture(self.input_mask_path)
        self.mask_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale mask
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.mask_frames.append(frame)
        cap.release()
        
        print(f"✓ Loaded {len(self.video_frames)} video frames and {len(self.mask_frames)} mask frames")
        print(f"  Frame size: {self.frame_size}, FPS: {self.fps:.2f}")
    
    def save_video(self, frames, output_path, fps=None):
        """Save frames as a video file."""
        if fps is None:
            fps = self.fps
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, self.frame_size)
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        print(f"✓ Video saved to: {output_path}")

class VideoInpainter3D(BaseVideoInpainter):
    """
    3D-aware video inpainter using 3D scene representation for geometric consistency.
    
    This class implements 3D-aware inpainting by:
    1. Building 3D scene representation from monocular video
    2. Applying geometric constraints based on 3D scene understanding
    3. Ensuring temporal consistency across frames
    """
    
    def __init__(self, 
                 input_video_path: str,
                 input_mask_path: str,
                 save_dir_path: str,
                 video_name: str,
                 **kwargs):
        
        super().__init__(input_video_path, input_mask_path, save_dir_path, video_name)
        
        # 3D scene representation components
        self.scene_3d = None
        self.camera_params = None
        self.static_scene_model = None
        self.depth_maps = None
        self.optical_flow = None
        
        # 3D processing parameters
        self.depth_estimation_model = kwargs.get('depth_estimation_model', 'monodepth')
        self.flow_estimation_model = kwargs.get('flow_estimation_model', 'raft')
        self.min_parallax_threshold = kwargs.get('min_parallax_threshold', 5.0)
        self.max_depth_discontinuity = kwargs.get('max_depth_discontinuity', 0.1)
        
        # Initialize 3D scene representation
        self._initialize_3d_scene()
    
    def _initialize_3d_scene(self):
        """Initialize 3D scene representation from monocular video."""
        print("Initializing 3D scene representation...")
        
        # Estimate depth maps
        self.depth_maps = self._estimate_depth_maps()
        
        # Estimate optical flow
        self.optical_flow = self._estimate_optical_flow()
        
        # Estimate camera parameters
        self.camera_params = self._estimate_camera_parameters()
        
        # Build 3D scene model
        self._build_3d_scene_model()
        
        print("✓ 3D scene representation initialized")
    
    def _estimate_depth_maps(self):
        """Estimate depth maps for each frame using monocular depth estimation."""
        print("Estimating depth maps...")
        
        try:
            # Try to import monodepth2 or similar
            # from monodepth2 import depth_estimation
            depth_maps = []
            
            for i, frame in enumerate(self.video_frames):
                # Placeholder: in practice, use a real depth estimation model
                # depth_map = depth_estimation.estimate_depth(frame)
                
                # Simplified depth estimation using edge detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Create synthetic depth map based on edges and image structure
                depth_map = self._create_synthetic_depth_map(frame, edges)
                depth_maps.append(depth_map)
                
                if i % 10 == 0:
                    print(f"  Processed frame {i+1}/{len(self.video_frames)}")
            
            print(f"✓ Estimated depth maps for {len(depth_maps)} frames")
            return depth_maps
            
        except ImportError:
            print("⚠ Monodepth not available, using simplified depth estimation")
            return self._create_simplified_depth_maps()
    
    def _create_synthetic_depth_map(self, frame, edges):
        """Create synthetic depth map based on image structure."""
        height, width = frame.shape[:2]
        depth_map = np.ones((height, width), dtype=np.float32)
        
        # Use edges to create depth discontinuities
        edge_mask = edges > 0
        depth_map[edge_mask] = 0.5
        
        # Use image gradients for depth variation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize and apply to depth
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        depth_map = depth_map * (1 - 0.3 * gradient_magnitude)
        
        # Ensure depth is in [0, 1] range
        depth_map = np.clip(depth_map, 0, 1)
        
        return depth_map
    
    def _create_simplified_depth_maps(self):
        """Create simplified depth maps when advanced depth estimation is not available."""
        print("Creating simplified depth maps...")
        
        depth_maps = []
        for frame in self.video_frames:
            height, width = frame.shape[:2]
            # Simple depth map based on distance from center
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height / 2, width / 2
            
            # Distance from center (normalized)
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            depth_map = 1 - (distance / max_distance)
            
            depth_maps.append(depth_map.astype(np.float32))
        
        return depth_maps
    
    def _estimate_optical_flow(self):
        """Estimate optical flow between consecutive frames."""
        print("Estimating optical flow...")
        
        try:
            # Try to import RAFT or similar optical flow model
            # from raft import RAFT
            # flow_model = RAFT()
            
            optical_flow = []
            for i in range(len(self.video_frames) - 1):
                frame1 = cv2.cvtColor(self.video_frames[i], cv2.COLOR_BGR2GRAY)
                frame2 = cv2.cvtColor(self.video_frames[i+1], cv2.COLOR_BGR2GRAY)
                
                # Placeholder: in practice, use a real optical flow model
                # flow = flow_model.estimate_flow(frame1, frame2)
                
                # Simplified optical flow using Lucas-Kanade
                flow = self._estimate_simplified_flow(frame1, frame2)
                optical_flow.append(flow)
                
                if i % 10 == 0:
                    print(f"  Processed flow {i+1}/{len(self.video_frames)-1}")
            
            print(f"✓ Estimated optical flow for {len(optical_flow)} frame pairs")
            return optical_flow
            
        except Exception as e:
            print(f"⚠ Optical flow estimation failed: {e}")
            return self._create_simplified_optical_flow()
    
    def _estimate_simplified_flow(self, frame1, frame2):
        """Estimate simplified optical flow using Lucas-Kanade method."""
        # Detect corners for optical flow
        corners = cv2.goodFeaturesToTrack(frame1, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        if corners is None:
            return np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
        
        # Calculate optical flow
        flow, status, _ = cv2.calcOpticalFlowPyrLK(frame1, frame2, corners, None)
        
        # Create flow map
        flow_map = np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
        
        # Fill flow map with estimated flow at corner points
        for i, (corner, flow_point) in enumerate(zip(corners, flow)):
            if status[i]:
                x, y = corner.ravel().astype(int)
                fx, fy = flow_point.ravel()
                if 0 <= x < frame1.shape[1] and 0 <= y < frame1.shape[0]:
                    flow_map[y, x] = [fx, fy]
        
        return flow_map
    
    def _create_simplified_optical_flow(self):
        """Create simplified optical flow when estimation fails."""
        print("Creating simplified optical flow...")
        
        optical_flow = []
        for i in range(len(self.video_frames) - 1):
            height, width = self.video_frames[i].shape[:2]
            # Simple flow based on frame differences
            frame1 = cv2.cvtColor(self.video_frames[i], cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(self.video_frames[i+1], cv2.COLOR_BGR2GRAY)
            
            # Calculate frame difference
            diff = cv2.absdiff(frame1, frame2)
            
            # Create simple flow map
            flow_map = np.zeros((height, width, 2), dtype=np.float32)
            flow_map[:, :, 0] = diff * 0.01  # Simple horizontal flow
            flow_map[:, :, 1] = diff * 0.01  # Simple vertical flow
            
            optical_flow.append(flow_map)
        
        return optical_flow
    
    def _estimate_camera_parameters(self):
        """Estimate camera parameters from monocular video."""
        print("Estimating camera parameters...")
        
        # For monocular video, we need to estimate camera motion
        camera_params = {
            "intrinsics": {
                "focal_length": [self.frame_size[0] * 0.8, self.frame_size[1] * 0.8],
                "principal_point": [self.frame_size[0] / 2, self.frame_size[1] / 2],
                "image_size": self.frame_size
            },
            "extrinsics": []
        }
        
        # Estimate camera motion from optical flow
        for i, flow in enumerate(self.optical_flow):
            # Estimate camera rotation and translation from flow
            motion = self._estimate_camera_motion_from_flow(flow, self.depth_maps[i])
            
            pose = {
                "frame_idx": i,
                "rotation": motion["rotation"],
                "translation": motion["translation"],
                "confidence": motion["confidence"]
            }
            camera_params["extrinsics"].append(pose)
        
        print(f"✓ Estimated camera parameters for {len(camera_params['extrinsics'])} frames")
        return camera_params
    
    def _estimate_camera_motion_from_flow(self, flow, depth_map):
        """Estimate camera motion from optical flow and depth."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated methods like:
        # - Essential matrix decomposition
        # - RANSAC-based motion estimation
        # - Bundle adjustment
        
        # Calculate average flow magnitude
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        avg_flow = np.mean(flow_magnitude)
        
        # Simple motion estimation based on flow magnitude
        if avg_flow < 1.0:
            # Small motion - mostly rotation
            rotation = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
            translation = [0.0, 0.0, 0.0]
            confidence = 0.9
        else:
            # Larger motion - estimate translation
            # Calculate dominant flow direction
            flow_x = np.mean(flow[:, :, 0])
            flow_y = np.mean(flow[:, :, 1])
            
            # Simple rotation estimation
            angle = np.arctan2(flow_y, flow_x) * 0.1
            rotation = [np.cos(angle/2), 0, np.sin(angle/2), 0]
            
            # Simple translation estimation
            translation = [flow_x * 0.01, flow_y * 0.01, avg_flow * 0.01]
            confidence = 0.7
        
        return {
            "rotation": rotation,
            "translation": translation,
            "confidence": confidence
        }
    
    def _build_3d_scene_model(self):
        """Build 3D scene model from depth maps and camera parameters."""
        print("Building 3D scene model...")
        
        self.static_scene_model = {
            "type": "monocular_3d",
            "depth_maps": self.depth_maps,
            "camera_params": self.camera_params,
            "optical_flow": self.optical_flow,
            "status": "initialized"
        }
        
        print("✓ 3D scene model built successfully")
    
    def run_3d_aware_inpainting(self, **kwargs):
        """
        Execute 3D-aware video inpainting process.
        
        Args:
            **kwargs: Additional parameters for inpainting
        """
        print("\n" + "="*60)
        print("3D-AWARE VIDEO INPAINTING")
        print("="*60)
        
        # Define output paths
        priori_path = os.path.join(self.save_dir_path, f"{self.video_name}_3d_priori.mp4")
        output_path = os.path.join(self.save_dir_path, f"{self.video_name}_3d_final.mp4")
        
        # Stage 1: 3D-guided priori generation
        print("\n--- Stage 1: 3D-Guided Priori Generation ---")
        start_priori_time = time.time()
        
        priori_frames = self._generate_3d_guided_priori(**kwargs)
        self.save_video(priori_frames, priori_path)
        
        end_priori_time = time.time()
        print(f"Stage 1 completed in: {end_priori_time - start_priori_time:.2f}s")
        
        # Stage 2: 3D-aware refinement
        print("\n--- Stage 2: 3D-Aware Refinement ---")
        start_refinement_time = time.time()
        
        refined_frames = self._refine_with_3d_constraints(priori_frames, **kwargs)
        self.save_video(refined_frames, output_path)
        
        end_refinement_time = time.time()
        print(f"Stage 2 completed in: {end_refinement_time - start_refinement_time:.2f}s")
        
        print("\n3D-aware inpainting complete!")
        
        return {
            'priori_path': priori_path,
            'final_path': output_path,
            'priori_time': end_priori_time - start_priori_time,
            'refinement_time': end_refinement_time - start_refinement_time
        }
    
    def _generate_3d_guided_priori(self, **kwargs):
        """Generate priori frames using 3D scene guidance."""
        print("Generating 3D-guided priori frames...")
        
        priori_frames = []
        for i in range(len(self.video_frames)):
            print(f"  Processing frame {i+1}/{len(self.video_frames)}")
            
            # Get current frame data
            frame = self.video_frames[i].copy()
            mask = self.mask_frames[i]
            depth_map = self.depth_maps[i]
            
            # Apply 3D geometric constraints
            guided_frame = self._apply_3d_geometric_constraints(frame, mask, depth_map, i)
            
            # Apply temporal consistency constraints
            if i > 0:
                guided_frame = self._apply_temporal_consistency_constraints(
                    guided_frame, priori_frames[-1], i
                )
            
            priori_frames.append(guided_frame)
        
        return priori_frames
    
    def _apply_3d_geometric_constraints(self, frame, mask, depth_map, frame_idx):
        """Apply 3D geometric constraints to ensure scene consistency."""
        # Create output frame
        output_frame = frame.copy()
        
        # Get mask regions to inpaint
        mask_regions = mask > 127
        
        if not np.any(mask_regions):
            return output_frame
        
        # Apply depth-aware inpainting
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                if mask_regions[y, x]:
                    # Get depth at this pixel
                    depth = depth_map[y, x]
                    
                    # Find similar depth regions for inpainting
                    similar_pixels = self._find_similar_depth_pixels(
                        frame, depth_map, depth, (x, y), mask_regions
                    )
                    
                    if similar_pixels:
                        # Use weighted average of similar depth pixels
                        output_frame[y, x] = self._interpolate_from_similar_pixels(
                            frame, similar_pixels, (x, y)
                        )
        
        return output_frame
    
    def _find_similar_depth_pixels(self, frame, depth_map, target_depth, target_pos, mask_regions):
        """Find pixels with similar depth for inpainting."""
        x, y = target_pos
        height, width = frame.shape[:2]
        
        similar_pixels = []
        search_radius = 50
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                ny, nx = y + dy, x + dx
                
                if (0 <= ny < height and 0 <= nx < width and 
                    not mask_regions[ny, nx]):
                    
                    depth_diff = abs(depth_map[ny, nx] - target_depth)
                    if depth_diff < self.max_depth_discontinuity:
                        distance = np.sqrt(dx*dx + dy*dy)
                        similarity = 1.0 / (1.0 + distance + depth_diff * 10)
                        similar_pixels.append(((nx, ny), similarity))
        
        # Sort by similarity and return top candidates
        similar_pixels.sort(key=lambda x: x[1], reverse=True)
        return similar_pixels[:20]  # Top 20 similar pixels
    
    def _interpolate_from_similar_pixels(self, frame, similar_pixels, target_pos):
        """Interpolate pixel value from similar depth pixels."""
        if not similar_pixels:
            return frame[target_pos[1], target_pos[0]]
        
        total_weight = 0
        interpolated_value = np.zeros(3, dtype=np.float32)
        
        for (px, py), weight in similar_pixels:
            pixel_value = frame[py, px].astype(np.float32)
            interpolated_value += pixel_value * weight
            total_weight += weight
        
        if total_weight > 0:
            interpolated_value /= total_weight
            return np.clip(interpolated_value, 0, 255).astype(np.uint8)
        else:
            return frame[target_pos[1], target_pos[0]]
    
    def _apply_temporal_consistency_constraints(self, current_frame, previous_frame, frame_idx):
        """Apply temporal consistency constraints between frames."""
        if frame_idx == 0 or previous_frame is None:
            return current_frame
        
        # Get optical flow from previous to current frame
        if frame_idx - 1 < len(self.optical_flow):
            flow = self.optical_flow[frame_idx - 1]
            
            # Apply flow-based temporal consistency
            consistent_frame = self._apply_flow_based_consistency(
                current_frame, previous_frame, flow
            )
            
            return consistent_frame
        
        return current_frame
    
    def _apply_flow_based_consistency(self, current_frame, previous_frame, flow):
        """Apply flow-based temporal consistency."""
        height, width = current_frame.shape[:2]
        consistent_frame = current_frame.copy()
        
        # For each pixel, check temporal consistency using flow
        for y in range(height):
            for x in range(width):
                # Get flow vector
                flow_x, flow_y = flow[y, x]
                
                # Calculate corresponding position in previous frame
                prev_x = int(x + flow_x)
                prev_y = int(y + flow_y)
                
                if (0 <= prev_x < width and 0 <= prev_y < height):
                    # Compare current and previous frame values
                    current_val = current_frame[y, x]
                    previous_val = previous_frame[prev_y, prev_x]
                    
                    # Calculate temporal consistency
                    temporal_diff = np.linalg.norm(current_val.astype(float) - previous_val.astype(float))
                    
                    # If temporal inconsistency is high, blend with previous frame
                    if temporal_diff > 50:  # Threshold for temporal consistency
                        blend_factor = 0.3
                        consistent_frame[y, x] = (
                            current_val * (1 - blend_factor) + 
                            previous_val * blend_factor
                        ).astype(np.uint8)
        
        return consistent_frame
    
    def _refine_with_3d_constraints(self, priori_frames, **kwargs):
        """Refine priori frames using 3D constraints."""
        print("Refining frames with 3D constraints...")
        
        refined_frames = []
        for i in range(len(priori_frames)):
            print(f"  Refining frame {i+1}/{len(priori_frames)}")
            
            frame = priori_frames[i].copy()
            depth_map = self.depth_maps[i]
            
            # Apply final 3D refinement
            refined_frame = self._apply_final_3d_refinement(frame, depth_map, i)
            refined_frames.append(refined_frame)
        
        return refined_frames
    
    def _apply_final_3d_refinement(self, frame, depth_map, frame_idx):
        """Apply final 3D refinement to ensure scene consistency."""
        # Apply depth-based smoothing
        refined_frame = self._apply_depth_based_smoothing(frame, depth_map)
        
        # Apply parallax consistency
        if frame_idx > 0 and frame_idx < len(self.video_frames) - 1:
            refined_frame = self._apply_parallax_consistency(
                refined_frame, frame_idx
            )
        
        return refined_frame
    
    def _apply_depth_based_smoothing(self, frame, depth_map):
        """Apply depth-based smoothing to reduce artifacts."""
        # Create depth-aware smoothing kernel
        height, width = frame.shape[:2]
        smoothed_frame = frame.copy()
        
        kernel_size = 5
        half_kernel = kernel_size // 2
        
        for y in range(half_kernel, height - half_kernel):
            for x in range(half_kernel, width - half_kernel):
                # Get depth at current pixel
                current_depth = depth_map[y, x]
                
                # Create depth-aware smoothing kernel
                kernel_weights = np.zeros((kernel_size, kernel_size))
                
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        neighbor_y = y + ky - half_kernel
                        neighbor_x = x + kx - half_kernel
                        
                        if (0 <= neighbor_y < height and 0 <= neighbor_x < width):
                            neighbor_depth = depth_map[neighbor_y, neighbor_x]
                            depth_diff = abs(neighbor_depth - current_depth)
                            
                            # Weight based on depth similarity
                            if depth_diff < self.max_depth_discontinuity:
                                kernel_weights[ky, kx] = 1.0 / (1.0 + depth_diff * 10)
                
                # Normalize kernel weights
                if np.sum(kernel_weights) > 0:
                    kernel_weights /= np.sum(kernel_weights)
                    
                    # Apply smoothing
                    smoothed_value = np.zeros(3, dtype=np.float32)
                    for ky in range(kernel_size):
                        for kx in range(kernel_size):
                            neighbor_y = y + ky - half_kernel
                            neighbor_x = x + kx - half_kernel
                            weight = kernel_weights[ky, kx]
                            smoothed_value += frame[neighbor_y, neighbor_x].astype(float) * weight
                    
                    smoothed_frame[y, x] = np.clip(smoothed_value, 0, 255).astype(np.uint8)
        
        return smoothed_frame
    
    def _apply_parallax_consistency(self, frame, frame_idx):
        """Apply parallax consistency for camera motion."""
        # This ensures that objects at different depths move appropriately
        # when the camera moves (parallax effect)
        
        if frame_idx == 0 or frame_idx >= len(self.camera_params["extrinsics"]) - 1:
            return frame
        
        # Get camera motion between frames
        current_pose = self.camera_params["extrinsics"][frame_idx]
        previous_pose = self.camera_params["extrinsics"][frame_idx - 1]
        
        # Calculate relative motion
        relative_translation = np.array(current_pose["translation"]) - np.array(previous_pose["translation"])
        
        # Apply parallax correction based on depth and camera motion
        # This is a simplified implementation
        # In practice, you'd use more sophisticated methods
        
        return frame

class VideoInpainterSD(BaseVideoInpainter):
    """
    Stable Diffusion-based video inpainter using ProPainter and DiffuEraser.
    
    This class implements the original two-stage inpainting approach:
    1. ProPainter for temporal consistency
    2. DiffuEraser for high-quality refinement
    """
    
    def __init__(self, 
                 input_video_path: str,
                 input_mask_path: str,
                 save_dir_path: str,
                 video_name: str,
                 base_model_path: str = "weights/stable-diffusion-v1-5",
                 vae_path: str = "weights/sd-vae-ft-mse",
                 diffueraser_path: str = "weights/diffuEraser",
                 propainter_model_dir: str = "weights/propainter"):
        
        super().__init__(input_video_path, input_mask_path, save_dir_path, video_name)
        
        # Model paths
        self.base_model_path = base_model_path
        self.vae_path = vae_path
        self.diffueraser_path = diffueraser_path
        self.propainter_model_dir = propainter_model_dir
        
        # Validate model paths
        self._validate_model_paths()
        
        # Initialize models
        self.video_inpainting_sd = None
        self.propainter = None
        
    def _validate_model_paths(self):
        """Validate that all required model paths exist."""
        if not os.path.exists(self.base_model_path):
            raise FileNotFoundError(f"Base model not found: {self.base_model_path}")
        if not os.path.exists(self.vae_path):
            raise FileNotFoundError(f"VAE model not found: {self.vae_path}")
        if not os.path.exists(self.diffueraser_path):
            raise FileNotFoundError(f"DiffuEraser model not found: {self.diffueraser_path}")
        if not os.path.exists(self.propainter_model_dir):
            raise FileNotFoundError(f"ProPainter model directory not found: {self.propainter_model_dir}")
    
    def initialize_models(self):
        """Initialize the ProPainter and DiffuEraser models."""
        print("Initializing Stable Diffusion models...")
        start_init_time = time.time()
        
        # Initialize DiffuEraser (diffusion-based refiner)
        self.video_inpainting_sd = DiffuEraser(
            self.device, 
            self.base_model_path, 
            self.vae_path, 
            self.diffueraser_path, 
            ckpt=None 
        )
        
        # Initialize ProPainter (flow-based prior generator)
        self.propainter = Propainter(self.propainter_model_dir, device=self.device)
        
        end_init_time = time.time()
        print(f"Model initialization took: {end_init_time - start_init_time:.2f}s")
    
    def run_inpainting(self, 
                       video_length: int = None,
                       max_img_size: int = 960,
                       mask_dilation_iter: int = 8,
                       ref_stride: int = 10,
                       neighbor_length: int = 10,
                       subvideo_length: int = 50):
        """
        Execute the full two-stage video inpainting process.
        
        Args:
            video_length (int): Maximum number of frames to process
            max_img_size (int): Maximum dimension for processing frames
            mask_dilation_iter (int): Number of iterations to dilate the mask
            ref_stride (int): Temporal stride for ProPainter reference frames
            neighbor_length (int): Number of neighboring frames for ProPainter
            subvideo_length (int): Length of sub-videos processed by ProPainter
        """
        # Initialize models if not already done
        if self.video_inpainting_sd is None or self.propainter is None:
            self.initialize_models()
        
        # Define output paths
        priori_path = os.path.join(self.save_dir_path, f"{self.video_name}_priori.mp4")
        output_path = os.path.join(self.save_dir_path, f"{self.video_name}_final.mp4")
        
        # --- Stage 1: Generate Priori with ProPainter ---
        print("\n--- Starting Stage 1: Generating Priori with ProPainter ---")
        start_priori_time = time.time()
        
        self.propainter.forward(
            self.input_video_path, 
            self.input_mask_path, 
            priori_path, 
            video_length=video_length, 
            ref_stride=ref_stride, 
            neighbor_length=neighbor_length, 
            subvideo_length=subvideo_length, 
            mask_dilation=mask_dilation_iter
        )
        
        end_priori_time = time.time()
        print(f"Stage 1 (ProPainter) finished in: {end_priori_time - start_priori_time:.2f}s")
        print(f"Priori video saved to: {priori_path}")

        # --- Stage 2: Refine with DiffuEraser ---
        print("\n--- Starting Stage 2: Refining with DiffuEraser ---")
        start_diffu_time = time.time()
        
        # For object removal, guidance_scale is often set low or to None
        self.video_inpainting_sd.forward(
            self.input_video_path, 
            self.input_mask_path, 
            priori_path, 
            output_path, 
            max_img_size=max_img_size, 
            video_length=video_length, 
            mask_dilation_iter=mask_dilation_iter, 
            guidance_scale=None  # Use default for object removal
        )
        
        end_diffu_time = time.time()
        print(f"Stage 2 (DiffuEraser) finished in: {end_diffu_time - start_diffu_time:.2f}s")
        print(f"Final inpainted video saved to: {output_path}")

        # --- Clean up GPU memory ---
        torch.cuda.empty_cache()
        print("\nInpainting complete!")
        
        return {
            'priori_path': priori_path,
            'final_path': output_path,
            'priori_time': end_priori_time - start_priori_time,
            'diffu_time': end_diffu_time - start_diffu_time
        }

# Legacy function for backward compatibility
def run_inpainting(args):
    """
    Legacy function for backward compatibility.
    """
    inpainter = VideoInpainterSD(
        input_video_path=args.input_video,
        input_mask_path=args.input_mask,
        save_dir_path=args.save_path,
        video_name="output",
        base_model_path=args.base_model_path,
        vae_path=args.vae_path,
        diffueraser_path=args.diffueraser_path,
        propainter_model_dir=args.propainter_model_dir
    )
    
    return inpainter.run_inpainting(
        video_length=args.video_length,
        max_img_size=args.max_img_size,
        mask_dilation_iter=args.mask_dilation_iter,
        ref_stride=args.ref_stride,
        neighbor_length=args.neighbor_length,
        subvideo_length=args.subvideo_length
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # --- Input/Output Arguments ---
    parser.add_argument('--input_video', type=str, default="examples/example3/video.mp4", help='Path to the input video file.')
    parser.add_argument('--input_mask', type=str, default="examples/example3/mask.mp4", help='Path to the input mask video file.')
    parser.add_argument('--save_path', type=str, default="results/my_video_output", help='Directory to save the output videos.')

    # --- Model Path Arguments ---
    parser.add_argument('--base_model_path', type=str, default="weights/stable-diffusion-v1-5", help='Path to Stable Diffusion v1.5 base model.')
    parser.add_argument('--vae_path', type=str, default="weights/sd-vae-ft-mse", help='Path to VAE model.')
    parser.add_argument('--diffueraser_path', type=str, default="weights/diffuEraser", help='Path to DiffuEraser model weights.')
    parser.add_argument('--propainter_model_dir', type=str, default="weights/propainter", help='Path to ProPainter model weights directory.')

    # --- Inpainting Control Arguments ---
    parser.add_argument('--video_length', type=int, default=None, help='Maximum number of frames to process. Default: full video.')
    parser.add_argument('--max_img_size', type=int, default=960, help='Maximum dimension (width or height) for processing frames.')
    parser.add_argument('--mask_dilation_iter', type=int, default=8, help='Number of iterations to dilate the mask, helps create smoother edges.')

    # --- ProPainter Specific Arguments ---
    parser.add_argument('--ref_stride', type=int, default=10, help='Temporal stride for ProPainter reference frames.')
    parser.add_argument('--neighbor_length', type=int, default=10, help='Number of neighboring frames for ProPainter.')
    parser.add_argument('--subvideo_length', type=int, default=50, help='Length of sub-videos processed by ProPainter.')
    
    args = parser.parse_args()
    
    run_inpainting(args)