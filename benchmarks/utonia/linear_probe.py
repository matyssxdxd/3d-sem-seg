import glob
import torch
import utonia
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F

from plyfile import PlyData
from torch.utils.data import Dataset, DataLoader

try:
    import flash_attn
    print("Using flash attention for faster inference.")
except ImportError:
    flash_attn = None

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
        }

        if self.transform:
            point = self.transform(point)
            point["segment"] = torch.as_tensor(segment, dtype=torch.long)

        return point

def restore_point_features(point):
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent

    feat = point.feat
    if "inverse" in point.keys():
        feat = feat[point.inverse]
    return feat

def load_model(device, num_classes):
    if flash_attn is not None:
        model = utonia.load("utonia", repo_id="Pointcept/Utonia").to(device)
    else:
        custom_config = dict(
            enc_patch_size=[4096 for _ in range(5)],
            enable_flash=False,
        )
        model = utonia.load(
            "utonia", repo_id="Pointcept/Utonia", custom_config=custom_config
        ).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dummy = {
        "coord": np.random.randn(4096, 3).astype(np.float32),
        "color": np.random.randn(4096, 3).astype(np.float32),
        "normal": np.zeros((4096, 3), dtype=np.float32),
        "segment": np.zeros(4096, dtype=np.float32)
    }
    dummy = transform(dummy)

    with torch.inference_mode():
        for k, v in dummy.items():
            if isinstance(v, torch.Tensor):
                dummy[k] = v.to(device)
        point = model(dummy)
        feat = restore_point_features(point)
        in_dim = feat.shape[-1]

    probe = nn.Linear(in_dim, num_classes).to(device)

    return model, probe

def train(model, probe, loader, optimizer, device):
    model.eval()
    probe.train()

    total_loss = 0.0
    total_pts = 0

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        target = batch["segment"].long()

        with torch.no_grad():
            point = model(batch)
            feat = restore_point_features(point)

        logits = probe(feat)
        loss = F.cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.numel()
        total_pts += target.numel()

    return total_loss / max(total_pts, 1)

def evaluate(model, probe, loader, device):
    model.eval()
    probe.eval()

    total_correct = 0
    total_valid = 0

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        target = batch["segment"].long()
        point = model(batch)
        feat = restore_point_features(point)
        logits = probe(feat)
        pred = logits.argmax(dim=-1)

        total_correct += (pred == target).sum().item()
        total_valid += target.numel() 

    return total_correct / max(total_valid, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", dest="data_path", help="path to LiDAR-Net data"
    )
    parser.add_argument(
        "--output_path", dest="output_path", help="path for checkpoint .pt file"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = args.data_path 
    num_classes = 30

    transform = utonia.transform.default(0.5)

    train_dataset = LiDARNetDataset(path=data_path, split="train", transform=transform)
    test_dataset = LiDARNetDataset(path=data_path, split="test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    model, probe = load_model(device, num_classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 20
    for epoch in range(epochs):
        train_loss = train(model, probe, train_loader, optimizer, device)
        val_acc = evaluate(model, probe, test_loader, device)
        print(f"epoch={epoch:03d}, train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

    torch.save({"probe": probe.state_dict()}, f"{args.output_path}/utonia_lidarnet.pt")
