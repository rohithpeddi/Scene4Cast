import argparse
import os

from torch.utils.data import DataLoader

from datasets.preprocess.reconstruction.base_4D_pipeline import Base4DPipeline, process_data, cuda_collate_fn


class AgMegaSAM4D(Base4DPipeline):
	
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
		
		self.megasam_root_dir = os.path.join(self.ag_root_directory, "megasam")
		self.megasam_4D_root_dir = os.path.join(self.ag_root_directory, "megasam_4D")
	
	def construct_video_4D_data(self, video_id):
		"""
		Constructs the 4D data from the video data.
		"""
		pass


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ag_root_directory", type=str, default=r"E:\DATA\OPEN\SceneGraphs\action_genome")
	parser.add_argument("--phase", type=str, default="train")
	parser.add_argument("--mode", type=str, default="predcls")
	parser.add_argument("--datasize", type=int, default=1000)
	parser.add_argument("--data_path", type=str, default=r"E:\DATA\OPEN\SceneGraphs\action_genome\cut3r")
	args = parser.parse_args()
	
	mode = args.mode
	data_path = args.data_path
	
	train_dataset = AgMegaSAM4D(
		phase="train",
		mode=mode,
		datasize="large",
		data_path=data_path,
		ag_root_directory=data_path,
		filter_nonperson_box_frame=True,
		filter_small_box=False if mode == 'predcls' else True
	)
	
	test_dataset = AgMegaSAM4D(
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