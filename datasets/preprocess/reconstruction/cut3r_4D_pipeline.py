import argparse
import os

import trimesh
import numpy as np
import torch.nn.functional as F
import torch
import numpy as np
import open3d as o3d
from collections import defaultdict
from PIL import Image
from torch.utils.data import DataLoader

from datasets.preprocess.reconstruction.base_4D_pipeline import Base4DPipeline, process_data
from datasets.preprocess.utils import load_file_pkl_file, load_torch_pkl_file, cuda_collate_fn, to_numpy
from datasets.preprocess.vis_utils import visualize_4d_point_clouds, visualize_4d_point_clouds_with_segmentation, \
	visualize_static_scene_with_segmentation


class AgCut3R4D(Base4DPipeline):
	
	def __init__(
			self,
			phase,
			mode,
			datasize,
			data_path,
			ag_root_directory,
			filter_nonperson_box_frame,
			filter_small_box
	):
		super().__init__(
			phase,
			mode,
			datasize,
			data_path,
			ag_root_directory,
			filter_nonperson_box_frame,
			filter_small_box
		)
		
		self.cut3r_root_dir = os.path.join(self.ag4D_root_dir, "cut3r")
		self.cut3r_4D_root_dir = os.path.join(self.ag4D_root_dir, "cut3r_4D")
		
		# Post-Processing Parameters
		self.voxel_size = 0.05  # Voxel size for discretizing space
		self.consensus_threshold = 0.7  # Threshold for static consensus
	
	def develop_static_scene(
			self,
			cut3r_video_data,
			masks_rescaled,
			dynamic_masks,
			video_id,
			timestamps,
	):
		video_static_scene_dir = os.path.join(self.cut3r_4D_root_dir, "static_scenes", video_id)
		
		pts3d = cut3r_video_data["pts3ds"]
		colors = cut3r_video_data["colors"]
		
		scene = trimesh.Scene()
		for pts, cols, t in zip(pts3d, colors, timestamps):
			loc_pts = pts.reshape(-1, 3)  # Reshape to (N, 3)
			loc_cols = cols.reshape(-1, 3).clone()  # Reshape to (N, 3)
			
			# Change the color of the points based on the segmentation mask such that
			# Those points can be clearly seen in the viewer.
			# For example, if the segmentation mask is 1, overlay a red color on the points.
			seg_mask = masks_rescaled[t].reshape(-1)  # (N,)
			bg = torch.tensor(seg_mask == 0)
			
			dyn_mask = dynamic_masks[t].reshape(-1)  # (N,)
			# Set the color to white for dynamic masks
			dyn_mask_bg = dyn_mask == 0
			
			# add the dynamic mask to the segmentation mask
			mask = torch.logical_or(bg, dyn_mask_bg)
			
			bg_pts = loc_pts[mask]
			bg_cols = loc_cols[mask]
			
			np_bg_pts = to_numpy(bg_pts)
			np_bg_cols = to_numpy(bg_cols)
			
			pct = trimesh.PointCloud(np_bg_pts, colors=np_bg_cols)
			scene.add_geometry(pct)
		
		# Show the processed scene
		scene.show()
		scene.export(file_obj=os.path.join(video_static_scene_dir, "static_scene.ply"))
	
	def construct_video_4D_data(self, video_id, enable_vis=False):
		"""
		Constructs the 4D data from the video data.
		"""
		cut3r_video_file_path = os.path.join(self.cut3r_root_dir, f"{video_id[:-4]}.pkl")
		cut3r_video_data = load_torch_pkl_file(cut3r_video_file_path)
		
		B, new_H, new_W, D = cut3r_video_data["pts3ds"][0].shape
		
		# -------------------------- PRE-PROCESSED DATA VISUALIZATIONS ----------------------
		# Visualize the CUT3R point cloud data.
		if enable_vis:
			# visualize_4D_point_clouds_open3D(
			# 	points=cut3r_video_data["pts3ds"],
			# 	colors=cut3r_video_data["colors"],
			# 	timestamps=list(range(len(cut3r_video_data["pts3ds"]))),
			# )
			
			# Rerun: Visualize the cut3r point cloud data using ReRun
			visualize_4d_point_clouds(
				point_clouds=cut3r_video_data["pts3ds"],
				colors=cut3r_video_data["colors"],
				timestamps=list(range(len(cut3r_video_data["pts3ds"]))),
			)
		
		# Load data from segmentation and cut3r pkl files.
		sam2_segmentation_file_path = os.path.join(self.processed_segmentation_root_dir, f"{video_id[:-4]}.pkl")
		sam2_video_masks = load_torch_pkl_file(sam2_segmentation_file_path)
		sam2_video_masks = sam2_video_masks.long()
		
		# 1a. Sam2 segmentation data is of the form (num_vide_frames, H, W)
		# Cut3r: video_id --> frame_ids [Each point cloud correspond to a single frame]
		frame_id_list = sorted(self.video_id_frame_id_list[video_id])
		frame_id_list = list(np.array(frame_id_list) - 1)  # Adjusting for 0-based indexing
		assert len(frame_id_list) == len(cut3r_video_data["pts3ds"])
		
		# Identify the indices of unique elements in the sorted frame_id_list
		unique_frame_ids = list(set(frame_id_list))
		unique_indices = [frame_id_list.index(frame_id) for frame_id in unique_frame_ids]
		
		# 1b. Load the dynamic masks output by monst3r
		dynamic_masks_file_path_list = []
		video_monst3r_dir = os.path.join(self.ag4D_root_dir, "monst3r", video_id)
		for file_name in os.listdir(video_monst3r_dir):
			if file_name.startswith("enlarged_dynamic_mask_") and file_name.endswith(".png"):
				dynamic_masks_file_path = os.path.join(video_monst3r_dir, file_name)
				dynamic_masks_file_path_list.append(dynamic_masks_file_path)
		assert len(dynamic_masks_file_path_list) == len(unique_frame_ids)
		
		# Load the png files and convert them to tensors
		dynamic_masks = []
		for mask_file_path in dynamic_masks_file_path_list:
			# Read PNG file from mask_file_path
			mask_image = np.array(Image.open(mask_file_path))
			mask_tensor = torch.from_numpy(mask_image).long()
			dynamic_masks.append(mask_tensor)
		
		# Convert the list of dynamic masks to a tensor
		dynamic_masks_tensor = torch.stack(dynamic_masks)
		
		# Check the shape of the dynamic masks tensor is same as new_H, new_W
		assert dynamic_masks_tensor.shape[1] == new_H and dynamic_masks_tensor.shape[2] == new_W
		
		# Remove non-unique elements from frame_id_list and corresponding point cloud data
		cut3r_video_data["pts3ds"] = [cut3r_video_data["pts3ds"][i] for i in unique_indices]
		
		# 2. Sample the masks corresponding to these frame ids
		unique_frame_ids_tensor = torch.Tensor(unique_frame_ids).long()
		masks_for_sampled_frames = sam2_video_masks[unique_frame_ids_tensor]
		assert masks_for_sampled_frames.shape[0] == len(cut3r_video_data["pts3ds"])
		
		# 3. Rescale the segmentation masks to fit the dimensions for the cut3r point cloud data.
		one_hot = F.one_hot(masks_for_sampled_frames, num_classes=self._num_obj_classes)
		one_hot = one_hot.permute(0, 3, 1, 2)  # (num_frames, num_classes, H, W)
		
		# Interpolate the masks to match the cut3r point cloud data dimensions all frames at once
		resized = F.interpolate(
			one_hot.float(),
			size=(new_H, new_W),
			mode="bilinear",
			align_corners=False,
		)
		
		# 4. Convert the resized masks back to the original shape
		masks_rescaled = resized.argmax(dim=1)  # (num_frames, new_H, new_W)
		timestamps = list(range(len(cut3r_video_data["pts3ds"])))
		
		# Visualize the CUT3R point cloud data along with SAM2 segmentations.
		if enable_vis:
			if masks_rescaled is not None and dynamic_masks_tensor is not None:
				assert masks_rescaled.shape == dynamic_masks_tensor.shape
			
			visualize_4d_point_clouds_with_segmentation(
				point_clouds=cut3r_video_data["pts3ds"],
				colors=cut3r_video_data["colors"],
				segmentation_masks=masks_rescaled,
				dynamic_masks=dynamic_masks_tensor,
				timestamps=timestamps,
				hsv_tuples=self.hsv_tuples,
				class_colors=self.class_colors
			)
		
		# --------------------------- STATIC SCENE CREATION ----------------------------------
		
		self.post_process_data(
			cut3r_video_data=cut3r_video_data,
			sam2_video_masks=masks_rescaled,
			dynamic_masks_tensor=dynamic_masks_tensor,
			video_id=video_id,
			timestamps=timestamps
		)
		
		visualize_static_scene_with_segmentation(
			point_clouds=cut3r_video_data["pts3ds"],
			colors=cut3r_video_data["colors"],
			timestamps=timestamps,
			segmentation_masks=masks_rescaled,
			dynamic_masks=dynamic_masks_tensor,
			app_id="static_scene_segmentation_demo"
		)
		
		# --------------------------- DYNAMIC DATA OVERLAY CREATION --------------------------
		
		return None
	
	def post_process_data(
			self,
			cut3r_video_data,
			sam2_video_masks,
			dynamic_masks_tensor,
			video_id,
			timestamps
	):
		# Combine both sam2 and dynamic masks
		assert sam2_video_masks.shape == dynamic_masks_tensor.shape
		aligned_masks = torch.logical_or(sam2_video_masks, dynamic_masks_tensor)
		
		# STEP 1
		static_voxels = self.consensus_static_scene(
			aligned_points=cut3r_video_data["pts3ds"],
			aligned_masks=aligned_masks,
			timestamps=timestamps
		)
		
		# STEP 2
		static_points, static_colors = self.extract_static_points(
			aligned_points=cut3r_video_data["pts3ds"],
			aligned_colors=cut3r_video_data["colors"],
			timestamps=timestamps,
			static_voxels=static_voxels
		)
		
		# STEP 3
		static_scene_pcd, static_scene_mesh = self.post_process_static_scene(static_points, static_colors)
		
		# Visualize the static scene
		scene = trimesh.Scene()
		scene.add_geometry(static_scene_mesh)
		scene.show()
	
	# Step 5: Consensus-based Static Scene Determination
	def consensus_static_scene(self, aligned_points, aligned_masks, timestamps):
		voxel_static_counts = defaultdict(int)
		voxel_total_counts = defaultdict(int)
		
		for pts, t in zip(aligned_points, timestamps):
			loc_pts = pts.reshape(-1, 3)  # Reshape to (N, 3)
			loc_pts = to_numpy(loc_pts)
			mask = aligned_masks[t].reshape(-1)  # (N,)
			mask = to_numpy(mask)
			static_points = loc_pts[mask == 0]
			
			# Compute voxel coordinates
			voxel_coords_all = np.floor(loc_pts / self.voxel_size).astype(int)
			voxel_coords_static = np.floor(static_points / self.voxel_size).astype(int)
			
			# Update counts
			for coord in map(tuple, voxel_coords_all):
				voxel_total_counts[coord] += 1
			for coord in map(tuple, voxel_coords_static):
				voxel_static_counts[coord] += 1
		
		# Compute consensus scores
		static_voxels = []
		for voxel in voxel_total_counts:
			total = voxel_total_counts[voxel]
			static_count = voxel_static_counts.get(voxel, 0)
			consensus_score = static_count / total
			
			if consensus_score >= self.consensus_threshold:
				static_voxels.append(voxel)
		
		print(f"Identified {len(static_voxels)} static voxels based on consensus.")
		
		return static_voxels
	
	# Step 6: Robust Static Extraction
	def extract_static_points(self, aligned_points, aligned_colors, timestamps, static_voxels):
		static_points_agg = []
		static_points_colors_agg = []
		static_voxel_set = set(static_voxels)
		for points, cols, t in zip(aligned_points, aligned_colors, timestamps):
			loc_pts = points.reshape(-1, 3)  # Reshape to (N, 3)
			loc_pts = to_numpy(loc_pts)
			
			loc_cols = cols.reshape(-1, 3)  # Reshape to (N, 3)
			loc_cols = to_numpy(loc_cols)
			
			voxel_coords = np.floor(loc_pts / self.voxel_size).astype(int)
			
			is_static_point = np.array([tuple(coord) in static_voxel_set for coord in voxel_coords])
			static_points = loc_pts[is_static_point]
			static_points_agg.append(static_points)
			static_points_colors = loc_cols[is_static_point]
			static_points_colors_agg.append(static_points_colors)
		
		# Aggregate all static points
		static_points_all = np.vstack(static_points_agg)
		static_points_colors_all = np.vstack(static_points_colors_agg)
		print(f"Total aggregated static points before filtering: {static_points_all.shape[0]}")
		return static_points_all, static_points_colors_all
	
	# Step 7: Post-processing and Final Static Scene Generation
	def post_process_static_scene(self, static_points, static_colors):
		scene = trimesh.Scene()
		
		# Create point cloud
		static_pcd = o3d.geometry.PointCloud()
		static_pcd.points = o3d.utility.Vector3dVector(static_points)
		static_pcd.colors = o3d.utility.Vector3dVector(static_colors)
		
		# Statistical outlier removal
		static_pcd, ind = static_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
		print(f"Points after statistical filtering: {np.asarray(static_pcd.points).shape[0]}")
		
		trimesh_pct = trimesh.PointCloud(static_points, colors=static_colors)
		scene.add_geometry(trimesh_pct)
		scene.show()
		
		# Optional: Surface Reconstruction using Poisson (for visualization)
		static_pcd.estimate_normals(
			search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2 * self.voxel_size, max_nn=30))
		mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(static_pcd, depth=9)
		
		# Remove low-density vertices to reduce noise in mesh
		densities = np.asarray(densities)
		density_threshold = np.quantile(densities, 0.05)
		vertices_to_remove = densities < density_threshold
		mesh.remove_vertices_by_mask(vertices_to_remove)
		mesh.compute_vertex_normals()
		print("Mesh reconstruction completed.")
		
		# Show the final output
		o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, window_name="Static Scene Mesh")
		
		return static_pcd, mesh


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ag_root_directory", type=str, default=r"E:\DATA\OPEN\SceneGraphs\action_genome")
	parser.add_argument("--phase", type=str, default="train")
	parser.add_argument("--mode", type=str, default="predcls")
	parser.add_argument("--datasize", type=int, default=1000)
	parser.add_argument("--data_path", type=str, default=r"E:\DATA\OPEN\SceneGraphs\action_genome")
	args = parser.parse_args()
	
	mode = args.mode
	data_path = args.data_path
	
	train_dataset = AgCut3R4D(
		phase="train",
		mode=mode,
		datasize="large",
		data_path=data_path,
		ag_root_directory=data_path,
		filter_nonperson_box_frame=True,
		filter_small_box=False if mode == 'predcls' else True
	)
	
	test_dataset = AgCut3R4D(
		phase="test",
		mode=mode,
		datasize="large",
		data_path=data_path,
		ag_root_directory=data_path,
		filter_nonperson_box_frame=True,
		filter_small_box=False if mode == 'predcls' else True
	)
	
	dataloader_train = DataLoader(
		train_dataset,
		shuffle=True,
		collate_fn=cuda_collate_fn,
		pin_memory=True,
		num_workers=0
	)
	
	dataloader_test = DataLoader(
		test_dataset,
		shuffle=True,
		collate_fn=cuda_collate_fn,
		pin_memory=False
	)
	
	# Train - Mode
	print("-----------------------------------------------------------------------")
	print(f"Processing Train - {mode} dataset")
	print("-----------------------------------------------------------------------")
	process_data(
		phase="train",
		mode=mode,
		dataloader=dataloader_train,
		data_path=data_path
	)
	
	# Test - Mode
	print("-----------------------------------------------------------------------")
	print(f"Processing Test - {mode} dataset")
	print("-----------------------------------------------------------------------")
	process_data(
		phase="test",
		mode=mode,
		dataloader=dataloader_test,
		data_path=data_path
	)


if __name__ == "__main__":
	main()
