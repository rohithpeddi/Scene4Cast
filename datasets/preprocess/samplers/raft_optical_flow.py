import os
from pathlib import Path

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
from tqdm import tqdm

plt.rcParams["savefig.bbox"] = "tight"


# sphinx_gallery_thumbnail_number = 2
def plot_flow_data(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.show()


class RAFTOpticalFlow:

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = raft_large(pretrained=True).to(self.device).eval()

        self.max_subsampled_idx = 200
        self.visualize = False

    @staticmethod
    def preprocess(batch):
        transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
                T.Resize(size=(520, 960)),
            ]
        )
        batch = transforms(batch)
        return batch

    def estimate_flow(self, image_batch1, image_batch2):
        assert image_batch1.shape == image_batch2.shape, "Input batches must have the same shape"
        assert image_batch1.dim() == 4 and image_batch1.size(1) == 3, "Input batches must be of shape (B, 3, H, W)"

        image_batch1 = image_batch1.to(self.device)
        image_batch2 = image_batch2.to(self.device)

        image_batch1 = self.preprocess(image_batch1)
        image_batch2 = self.preprocess(image_batch2)

        with torch.no_grad():
            list_of_flows = self.model(image_batch1, image_batch2)

        return list_of_flows[-1]

    @staticmethod
    def build_image_batch(frame_files):
        image_list = []
        for frame_file_path in frame_files:
            # Load image from the path and convert to tensor
            img = Image.open(frame_file_path).convert("RGB")
            img_tensor = F.pil_to_tensor(img)
            image_list.append(img_tensor)
        return torch.stack(image_list, dim=0)

    def init_flow_estimation_for_video(
            self,
            video_id: str,
            threshold: float = 0.5,  # change threshold on p95 flow magnitude
            max_samples: int = 150,  # hard cap on number of frames to keep
            probe_batch: int = 8  # how many future candidates to test at once
    ):
        video_frames_dir = self.data_dir / "frames" / video_id
        frame_files = sorted(video_frames_dir.glob("*.png"))

        if len(frame_files) == 0:
            print(f"[{video_id}] No frames found at {video_frames_dir}")
            return []
        if len(frame_files) == 1:
            print(f"[{video_id}] Only one frame found; returning [0]")
            return [0]

        # Load all frames once
        imgs = self.build_image_batch(frame_files)  # (N, C, H, W)
        N = imgs.size(0)

        selected = [0]  # always keep the first frame
        last_idx = 0
        cand = 1

        with torch.inference_mode():
            pbar = tqdm(total=N - 1, desc=f"Processing {video_id}", leave=False)
            while cand < N and len(selected) < max_samples:
                end = min(cand + probe_batch, N)
                cand_imgs = imgs[cand:end]  # (B,C,H,W)
                last_rep = imgs[last_idx].unsqueeze(0).expand_as(cand_imgs)

                flow = self.estimate_flow(last_rep, cand_imgs)  # (B,2,H,W)
                mag = torch.norm(flow, dim=1)  # (B,H,W)
                scores = torch.quantile(mag.flatten(1), 0.95, dim=1)  # robust motion score per candidate

                meets = (scores >= threshold).nonzero(as_tuple=True)[0]
                if meets.numel() > 0:
                    rel = int(meets[0].item())
                    new_idx = cand + rel
                    selected.append(new_idx)
                    last_idx = new_idx
                    cand = last_idx + 1
                else:
                    cand = end  # advance window

                pbar.update(end - pbar.n)
            pbar.close()

        print("---------------------------------------------------------------------------------------------")
        print(f"[{video_id}] Selected {len(selected)} frames (cap {max_samples}) with threshold {threshold}")

        # Include all the frames annotated into the selected list
        annotated_frames_dir = self.data_dir / "frames_annotated" / video_id
        annotated_frame_files = sorted(annotated_frames_dir.glob("*.png"))
        annotated_frame_indices = {int(f.stem) for f in annotated_frame_files if f.stem.isdigit()}
        selected_set = set(selected)
        keep_idx = selected_set.union(annotated_frame_indices)
        selected = sorted(keep_idx)

        print(f"[{video_id}] After including annotated frames, total selected frames: {len(selected)}")
        return selected


def main():
    data_dir = Path("/data/rohith/ag")
    raft_of = RAFTOpticalFlow(data_dir)

    video_id_list = os.listdir(data_dir / "videos")
    output_dir = data_dir / "raft_optical_flow"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the sampled frame indices npy file for each video_id
    video_id_frame_idx_dict = {}
    for video_id in tqdm(video_id_list):
        sub_sampled_idx = raft_of.init_flow_estimation_for_video(video_id)
        # Store the result in the dictionary and also save as a npy file
        np.save(output_dir / f"{video_id[:-4]}.npy", np.array(sub_sampled_idx))
        video_id_frame_idx_dict[video_id] = sub_sampled_idx

    # Save the video_id_frame_idx_dict as a pkl file
    with open(data_dir / "video_id_frame_idx_dict_raft.pkl", "wb") as f:
        pickle.dump(video_id_frame_idx_dict, f)


if __name__ == "__main__":
    main()
