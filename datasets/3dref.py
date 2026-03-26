import numpy as np
import glob

from torch.utils.data import Dataset
from plyfile import PlyData
from pathlib import Path
from typing import Dict

LABEL_INFO = {
    0: ("Unlabeled", (255, 255, 255)),                  # White
    1: ("Normal Points", (128, 128, 128)),              # Gray
    2: ("Glass", (0, 255, 0)),                          # Green
    3: ("Mirrors", (0, 0, 139)),                        # Dark blue
    4: ("Other reflective objects", (173, 216, 230)),   # Light blue
    5: ("Reflection points", (255, 105, 180)),          # Pink
    6: ("Obstacle behind glass", (255, 255, 0)),        # Yellow
}

class ThreeDeeRefDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.path = Path(path)

        self.files = sorted(glob.glob(str(self.path / "*.ply")))
        if not self.files:
            raise RuntimeError(f"No .ply files found under {self.path}")


    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict:
        ply = PlyData.read(self.files[index])
        v = ply["vertex"].data

        coord = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
        intensity = np.asarray(v["intensity"], dtype=np.float32)
        label = np.asarray(v["label"], dtype=np.int64)

        point = {
            "coord": coord,
            "intensity": intensity,
            "label": label
        }

        return point

if __name__ == "__main__":
    dataset = ThreeDeeRefDataset("/home/matyss/Downloads/3dref/raw/seq1/raycast/pointcloud/hesai/")
    print(dataset[0])

