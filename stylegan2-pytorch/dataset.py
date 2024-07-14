import os
import random

from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, shuffle=True):
        
        self.img_paths = []
        
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".jpg"):
                    self.img_paths.append(os.path.join(root, file))
        
        if shuffle:
            random.shuffle(self.img_paths)

        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert("RGB")
        img = self.transform(img)
        return img
