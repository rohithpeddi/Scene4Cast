import os

import argparse

from tqdm import tqdm


class MegaSamProcessor:
	
	def __init__(
			self,
			mega_sam_root_directory,
			compressed_mega_sam_root_directory,
	):
		self.mega_sam_root_directory = mega_sam_root_directory
		self.compressed_mega_sam_root_directory = compressed_mega_sam_root_directory
		os.makedirs(self.compressed_mega_sam_root_directory, exist_ok=True)
	
	@staticmethod
	def process_video_directories(uncomp_video_dir_root, comp_video_dir_root, stage_name):
		# List all the video_directories in uncomp reconstructions folder
		video_directories = os.listdir(uncomp_video_dir_root)
		# For each video directory, check if it already exists in the compressed reconstructions folder
		# If not compress it to a zip file and copy it to the compressed reconstructions folder
		for video_directory in tqdm(video_directories, desc=f"Compressing {stage_name} video directories"):
			video_directory_path = os.path.join(uncomp_video_dir_root, video_directory)
			if os.path.isdir(video_directory_path):
				# Check if the compressed file already exists
				compressed_file_path = os.path.join(comp_video_dir_root, f"{video_directory[:-4]}.zip")
				if not os.path.exists(compressed_file_path):
					# Compress the directory to a zip file
					os.system(f"zip -r {compressed_file_path} {video_directory_path}")
					print(f"Compressed {video_directory} to {compressed_file_path}")
				else:
					print(f"Compressed file for {video_directory} already exists.")
	
	def compress_data(self):
		# 1. Compress camera_tracking data
		uncomp_camera_tracking_path = os.path.join(self.mega_sam_root_directory, "camera_tracking")
		comp_camera_tracking_path = os.path.join(self.compressed_mega_sam_root_directory, "camera_tracking")
		os.makedirs(comp_camera_tracking_path, exist_ok=True)
		
		uncomp_reconstruction_path = os.path.join(uncomp_camera_tracking_path, "reconstructions")
		comp_reconstruction_path = os.path.join(comp_camera_tracking_path, "reconstructions")
		os.makedirs(comp_reconstruction_path, exist_ok=True)
		
		self.process_video_directories(
			uncomp_video_dir_root=uncomp_reconstruction_path,
			comp_video_dir_root=comp_reconstruction_path,
			stage_name="camera_tracking"
		)
		
		# 2. Compress depth anything data
		uncomp_depth_anything_root_dir = os.path.join(self.mega_sam_root_directory, "depth_anything")
		comp_depth_anything_root_dir = os.path.join(self.compressed_mega_sam_root_directory, "depth_anything")
		os.makedirs(comp_depth_anything_root_dir, exist_ok=True)
		
		self.process_video_directories(
			uncomp_video_dir_root=uncomp_depth_anything_root_dir,
			comp_video_dir_root=comp_depth_anything_root_dir,
			stage_name="depth_anything"
		)
		
		# 3. Compress unidepth data
		
		uncomp_unidepth_root_dir = os.path.join(self.mega_sam_root_directory, "unidepth")
		comp_unidepth_root_dir = os.path.join(self.compressed_mega_sam_root_directory, "unidepth")
		os.makedirs(comp_unidepth_root_dir, exist_ok=True)
		self.process_video_directories(
			uncomp_video_dir_root=uncomp_unidepth_root_dir,
			comp_video_dir_root=comp_unidepth_root_dir,
			stage_name="unidepth"
		)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mega_sam_root_directory", type=str, default="/data/rohith/ag/ag4D/mega_sam")
	parser.add_argument("--compressed_mega_sam_root_directory", type=str,
	                    default="/data/rohith/ag/ag4D/compressed_mega_sam")
	args = parser.parse_args()
	
	mega_sam_processor = MegaSamProcessor(
		mega_sam_root_directory=args.mega_sam_root_directory,
		compressed_mega_sam_root_directory=args.compressed_mega_sam_root_directory
	)
	
	mega_sam_processor.compress_data()


if __name__ == "__main__":
	main()
