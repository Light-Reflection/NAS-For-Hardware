"""
Make dataset loader for gan module
"""
from torch.utils.data import Dataset
import os
import torch
from skimage import io, transform

class LFWDataset(Dataset):
    """lfw aligned dataset"""
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.imgs = os.listdir(root_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.imgs[idx])
        img = io.imread(img_path)
        if self.transforms:
            img = self.transforms(img)
        return img






