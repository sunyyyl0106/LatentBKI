import os
import cv2
import numpy as np
import warnings
from torch.utils.data import Dataset
import yaml

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

config_file = os.path.join(os.getcwd(), "ipca.yaml")
config = yaml.safe_load(open(config_file, 'r'))

class HabitatDataset(Dataset):
    """Dataset class for mp3d habitat."""

    def __init__(self, dataset_dir, data_split='train'):
        """Read in the data from disk."""
        super().__init__()

        if data_split not in ["train", "val", "all"]:
            raise ValueError("Partition {} does not exist".format(data_split))

        self.split = data_split
        if data_split == "all":
            self.seq_names = (
                    config['split']['train'] + config['split']['val'] 
            )
        else:
            self.seq_names = config['split'][self.split]

        self.seq_dirs = [os.path.join(dataset_dir, sequence) for sequence in self.seq_names]
        self.seqs = [HabitatFrameDataset(seq_dir, self.seq_names[i]) for i, seq_dir in enumerate(self.seq_dirs)]

    def __len__(self):
        """Return size of dataset."""
        return len(self.seqs)

    def __getitem__(self, idx):
        """Return sample at index `idx` of dataset."""
        return self.seqs[idx]


class HabitatFrameDataset(Dataset):
    """Dataset class for sequences of scenes."""

    def __init__(self, seq_dir, seq_name):
        """Read data from disk for a single scene."""
        super().__init__()
        # Define paths to the dir
        self.seq_name = seq_name
        self.dir = seq_dir
        self.rgb_dir = os.path.join(seq_dir, 'rgb')
        self.depth_dir = os.path.join(seq_dir, 'depth')
        self.semantic_dir = os.path.join(seq_dir, 'semantic')
        self.pose_file = os.path.join(seq_dir, 'poses.txt')
        # Define paths to the scene
        self.rgb_files = [os.path.join(self.rgb_dir, file) for file in sorted(os.listdir(self.rgb_dir))]
        self.depth_files = [os.path.join(self.depth_dir, file) for file in sorted(os.listdir(self.depth_dir))]
        self.semantic_files = [os.path.join(self.semantic_dir, file)
                               for file in sorted(os.listdir(self.semantic_dir))]
        self.poses = [line for line in np.loadtxt(self.pose_file)]

    def __len__(self):
        """Return the number of frames in the scene."""
        return len(self.rgb_files)

    def __getitem__(self, idx):
        """Return data for a single frame."""
        # Load the RGB image
        bgr = cv2.imread(self.rgb_files[idx])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Load the depth
        with open(self.depth_files[idx], "rb") as f:
            depth = np.load(f)
        # Load the ground-truth semantic map
        semantic = np.load(self.semantic_files[idx])
        # Get the pose
        pose = self.poses[idx]
        frame = {'rgb': rgb, 'depth': depth, 'pose': pose, 'semantic': semantic}
        return frame










