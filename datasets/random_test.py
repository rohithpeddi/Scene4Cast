import json
import pickle


def convert_pkl_to_json():
	pkl_file_path = r"C:\Users\rohit\PycharmProjects\Scene4Cast\datasets\4d_video_frame_id_list.pkl"
	json_file_path = r"C:\Users\rohit\PycharmProjects\Scene4Cast\datasets\4d_video_frame_id_list.json"
	with open(pkl_file_path, 'rb') as pkl_file:
		data = pickle.load(pkl_file)
		
	sorted_unique_data = {}
	for key, value in data.items():
		sorted_unique_data[key] = sorted(set(value))
	
	with open(json_file_path, 'w') as json_file:
		json.dump(sorted_unique_data, json_file)


if __name__ == "__main__":
	convert_pkl_to_json()
