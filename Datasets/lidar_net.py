import torch
import glob
import numpy as np
from pathlib import Path

from plyfile import PlyData
from torch.utils.data import Dataset

class LiDARNetDataset(Dataset):
    def __init__(self, path, split="train", transform=None):
        self.path = Path(path)

        self.files = sorted(glob.glob(str(self.path / split / "*.ply")))
        if not self.files:
            raise RuntimeError(f"No .ply files found under {self.path / split}")

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        ply = PlyData.read(self.files[index])
        v = ply["vertex"].data

        coord = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
        color = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)

        if color.max() > 1.0:
            color = color / 255.0

        segment = np.asarray(v["sem"], dtype=np.int64)

        point = {"coord": coord,
                 "color": color,
                 "normal": np.zeros_like(coord, dtype=np.float32),
                 "segment": segment }

        if self.transform:
            point = self.transform(point)

        return point

if __name__ == "__main__":
    dataset = LiDARNetDataset("/home/matyss/Masters/data/LiDAR_Net/") 
    print(dataset[0])


