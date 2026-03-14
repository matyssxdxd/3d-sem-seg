import glob
import numpy as np
from pathlib import Path

from plyfile import PlyData
from torch.utils.data import Dataset

LIDAR_NET_IDS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30
)

CLASS_LABELS = (
    "unkown",
    "houseplant",
    "tree",
    "person",
    "floor",
    "stair",
    "ceiling",
    "pipe",
    "wall",
    "pillar",
    "window",
    "curtain",
    "door",
    "table",
    "chair",
    "sofa",
    "blackboard",
    "monitor",
    "bookshelf",
    "wardrobe",
    "bed",
    "reflection noise",
    "ghost",
    "light",
    "tabletop others"
)

LIDAR_NET_COLORMAP = {
    0:  (0.0, 0.0, 0.0),
    1:  (0.2, 0.8, 0.2),
    2:  (0.0, 0.6, 0.0),
    3:  (1.0, 0.2, 0.2),
    4:  (0.6, 0.6, 0.6),
    5:  (0.8, 0.5, 0.2),
    6:  (0.9, 0.9, 0.9),
    7:  (0.2, 0.7, 0.7),
    8:  (0.5, 0.5, 0.7),
    9:  (0.7, 0.4, 0.2),
    10: (0.4, 0.8, 1.0),
    11: (0.8, 0.4, 0.8),
    12: (0.6, 0.3, 0.1),
    13: (0.9, 0.7, 0.3),
    14: (0.3, 0.6, 0.9),
    15: (0.7, 0.2, 0.5),
    16: (0.1, 0.1, 0.1),
    17: (0.2, 0.9, 0.9),
    18: (0.4, 0.2, 0.6),
    19: (0.5, 0.3, 0.4),
    20: (0.9, 0.6, 0.6),
    21: (1.0, 0.0, 1.0),
    22: (0.6, 0.0, 0.6),
    23: (1.0, 1.0, 0.3),
    24: (0.8, 0.8, 0.4),
    25: (0.3, 0.3, 0.3),
    26: (0.0, 0.4, 0.8),
    27: (0.8, 0.2, 0.0),
    28: (0.2, 0.8, 0.6),
    29: (0.6, 0.2, 0.8),
    30: (0.8, 0.8, 0.8)
}

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
    print(len(dataset))
    print(min(dataset[0]["segment"]), max(dataset[0]["segment"]))
    print(min(dataset[1]["segment"]), max(dataset[1]["segment"]))
    print(min(dataset[2]["segment"]), max(dataset[2]["segment"]))
    print(min(dataset[3]["segment"]), max(dataset[3]["segment"]))


