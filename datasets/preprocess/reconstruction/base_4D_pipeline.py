import colorsys
import os
import pickle
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from constants import Constants as const
from datasets.preprocess.utils import save_dict_to_pkl


class Base4DPipeline(Dataset):

    def __init__(
            self,
            phase,
            mode,
            datasize,
            data_path,
            ag_root_directory,
            filter_nonperson_box_frame=False,
            filter_small_box=False
    ):
        self.ag_root_directory = ag_root_directory

        self.segmentation_root_dir = os.path.join(self.ag_root_directory, "segmentation")
        self.processed_segmentation_root_dir = os.path.join(self.segmentation_root_dir, "processed")
        self.ag4D_root_dir = os.path.join(self.ag_root_directory, "ag4D")

        self._phase = phase
        self._mode = mode
        self._datasize = datasize
        self._data_path = data_path
        self._frames_path = os.path.join(self._data_path, "frames")

        self._fetch_object_classes()

        # Fetch object and person bounding boxes
        person_bbox, object_bbox = self._fetch_object_person_bboxes(self._datasize, filter_small_box)

        # collect valid frames
        video_dict, q = self._fetch_valid_frames(person_bbox, object_bbox)
        all_video_names = np.unique(q)

        # Build dataset
        # self._build_dataset(video_dict, person_bbox, object_bbox, all_video_names, filter_nonperson_box_frame)
        self._video_list = [["0A8CF.mp4/1"]]
        
        # For each video_id, frame_names start from 1
        # So for all tensor calculations we need to subtract 1 from the frame names to subsample the data
        video_id_frame_id_list_pkl_file_path = os.path.join(os.path.dirname(__file__), "../../4d_video_frame_id_list.pkl")
        if os.path.exists(video_id_frame_id_list_pkl_file_path):
            with open(video_id_frame_id_list_pkl_file_path, "rb") as f:
                self.video_id_frame_id_list = pickle.load(f)
        else:
            assert False, f"Please generate {video_id_frame_id_list_pkl_file_path} first"
            
        self._num_obj_classes = len(self.object_classes)
        
        self.hsv_tuples = [(i / 37.0, 1.0, 1.0) for i in range(37)]
        self.class_colors = np.array([
            colorsys.hsv_to_rgb(h, s, v) for (h, s, v) in self.hsv_tuples
        ], dtype=np.float32)  # shape (37,3), values in [0,1]

    @staticmethod
    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def _fetch_object_classes(self):
        self.object_classes = [const.BACKGROUND]
        with open(os.path.join(self._data_path, const.ANNOTATIONS, const.OBJECT_CLASSES_FILE), 'r',
                  encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

    def _fetch_object_person_bboxes(self, datasize, filter_small_box=False):
        annotations_path = os.path.join(self._data_path, const.ANNOTATIONS)
        if filter_small_box:
            with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_FILTERSMALL_PKL),
                      'rb') as f:
                object_bbox = pickle.load(f)
        else:
            with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL), 'rb') as f:
                object_bbox = pickle.load(f)
            f.close()
        return person_bbox, object_bbox

    def _fetch_valid_frames(self, person_bbox, object_bbox):
        video_dict = {}
        q = []
        for i in person_bbox.keys():
            if object_bbox[i][0][const.METADATA][const.SET] == self._phase:  # train or testing?
                video_name, frame_num = i.split('/')
                q.append(video_name)
                frame_valid = False
                for j in object_bbox[i]:  # the frame is valid if there is visible bbox
                    if j[const.VISIBLE]:
                        frame_valid = True
                if frame_valid:
                    video_name, frame_num = i.split('/')
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(i)
                    else:
                        video_dict[video_name] = [i]
        return video_dict, q

    def _build_dataset(self, video_dict, person_bbox, object_bbox, all_video_names, filter_nonperson_box_frame=True):
        self._valid_video_names = []
        self._video_list = []
        self._video_size = []  # (w,h)
        self._gt_annotations = []
        self._non_gt_human_nums = 0
        self._non_heatmap_nums = 0
        self._non_person_video = 0
        self._one_frame_video = 0
        self._valid_nums = 0
        self._invalid_videos = []

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''
        for i in video_dict.keys():
            video = []
            gt_annotation_video = []
            for j in video_dict[i]:
                if filter_nonperson_box_frame:
                    if person_bbox[j][const.BOUNDING_BOX].shape[0] == 0:
                        self._non_gt_human_nums += 1
                        continue
                    else:
                        video.append(j)
                        self._valid_nums += 1

                gt_annotation_frame = [
                    {
                        const.PERSON_BOUNDING_BOX: person_bbox[j][const.BOUNDING_BOX],
                        const.FRAME: j
                    }
                ]

                # each frame's objects and human
                for k in object_bbox[j]:
                    if k[const.VISIBLE]:
                        assert k[const.BOUNDING_BOX] is not None, 'warning! The object is visible without bbox'
                        k[const.CLASS] = self.object_classes.index(k[const.CLASS])
                        # from xywh to xyxy
                        k[const.BOUNDING_BOX] = np.array([
                            k[const.BOUNDING_BOX][0], k[const.BOUNDING_BOX][1],
                            k[const.BOUNDING_BOX][0] + k[const.BOUNDING_BOX][2],
                            k[const.BOUNDING_BOX][1] + k[const.BOUNDING_BOX][3]
                        ])
                        gt_annotation_frame.append(k)
                gt_annotation_video.append(gt_annotation_frame)

            if len(video) > 2:
                self._video_list.append(video)
                self._video_size.append(person_bbox[j][const.BOUNDING_BOX_SIZE])
                self._gt_annotations.append(gt_annotation_video)
            elif len(video) == 1:
                self._one_frame_video += 1
            else:
                self._non_person_video += 1

        print('x' * 60)
        if filter_nonperson_box_frame:
            print('There are {} videos and {} valid frames'.format(len(self._video_list), self._valid_nums))
            print('{} videos are invalid (no person), remove them'.format(self._non_person_video))
            print('{} videos are invalid (only one frame), remove them'.format(self._one_frame_video))
            print('{} frames have no human bbox in GT, remove them!'.format(self._non_gt_human_nums))
        else:
            print('There are {} videos and {} valid frames'.format(len(self._video_list), self._valid_nums))
            print('{} frames have no human bbox in GT'.format(self._non_gt_human_nums))
            print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(
                self._non_heatmap_nums))
        print('x' * 60)

        self.invalid_video_names = np.setdiff1d(all_video_names, self._valid_video_names, assume_unique=False)

    def __len__(self):
        return len(self._video_list)

    @abstractmethod
    def construct_video_4D_data(self, video_id, enable_vis=False):
        """
        Constructs the 4D data from the video data.
        """
        pass

    def __getitem__(self, index):
        """
        Returns the 4D data for the given item.
        """
        frame_names = self._video_list[index]
        # gt_annotation_frame = self._gt_annotations[index]
        # file_directory_path = f"{self._data_path}/segmentation/{self._phase}/{self._mode}"
        video_id = frame_names[0].split('/')[0]
        if video_id != "0A8CF.mp4":
            return None
        video_4D_data = self.construct_video_4D_data(video_id, enable_vis=True)
        return video_4D_data


def process_data(phase, mode, dataloader, data_path):
    file_directory_path = f"{data_path}/segmentation/{phase}/{mode}"
    if not os.path.exists(file_directory_path):
        os.makedirs(file_directory_path)

    for i, video_segments in enumerate(tqdm(dataloader, desc="Training Progress")):

        if video_segments is None:
            print("Skipping video as it is not part of the split or already processed.")
            continue

        video_id = video_segments["video_id"]
        file_path = os.path.join(file_directory_path, f"{video_id}.pkl")
        save_dict_to_pkl(video_segments, file_path)