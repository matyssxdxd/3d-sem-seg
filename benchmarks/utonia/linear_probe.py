import os
import glob
import torch
import utonia
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F

from plyfile import PlyData
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import flash_attn
    print("Using flash attention for faster inference.")
except ImportError:
    flash_attn = None

RAW_CLASS_INFO = [
    ("Houseplant", 3),
    ("Tree", 4),
    ("Person", 5),
    ("Floor", 9),
    ("Stair", 10),
    ("Ceiling", 11),
    ("Pipe", 12),
    ("Wall", 13),
    ("Pillar", 14),
    ("Window", 15),
    ("Curtain", 16),
    ("Door", 17),
    ("Table", 18),
    ("Chair", 19),
    ("Sofa", 20),
    ("Blackboard", 21),
    ("Monitor", 22),
    ("Bookshelf", 23),
    ("Wardrobe", 24),
    ("Bed", 25),
    ("Reflection noise", 26),
    ("Ghost", 27),
    ("Light", 29),
    ("Tabletop others", 30),
]

CLASS_NAMES = [name for name, _ in RAW_CLASS_INFO]
RAW_CLASS_IDS = [raw_id for _, raw_id in RAW_CLASS_INFO]

IGNORE_INDEX = -1
NUM_CLASSES = len(RAW_CLASS_IDS)   # 24
MAX_RAW_LABEL = max(RAW_CLASS_IDS + [0])  # 30

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

        max_points = 100000
        if coord.shape[0] > max_points:
            idx = np.random.choice(coord.shape[0], max_points, replace=False)
            coord = coord[idx]
            color = color[idx]
            segment = segment[idx]

        point = {
            "coord": coord,
            "color": color,
            "normal": np.zeros_like(coord, dtype=np.float32),
            "segment": segment,
        }

        if self.transform:
            point = self.transform(point)

        point["segment"] = torch.as_tensor(segment, dtype=torch.long)

        return point

def build_label_map():
    """
    Maps raw semantic IDs -> contiguous train IDs.
    Raw label 0 (unannotated) and any unknown raw labels map to IGNORE_INDEX.
    """
    label_map = torch.full((MAX_RAW_LABEL + 1,), IGNORE_INDEX, dtype=torch.long)
    for new_id, raw_id in enumerate(RAW_CLASS_IDS):
        label_map[raw_id] = new_id
    return label_map


def remap_labels(raw_target, label_map):
    """
    raw_target: tensor of raw semantic labels from dataset
    returns: tensor with values in [0, NUM_CLASSES-1] or IGNORE_INDEX
    """
    raw_target = raw_target.long()
    target = torch.full_like(raw_target, IGNORE_INDEX)

    valid = (raw_target >= 0) & (raw_target < label_map.numel())
    target[valid] = label_map[raw_target[valid]]
    return target

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
            enc_patch_size=[1024 for _ in range(5)],
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

def train(model, probe, loader, optimizer, device, label_map):
    model.eval()
    probe.train()

    total_loss = 0.0
    total_pts = 0

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        raw_target = batch["segment"].long()
        target = remap_labels(raw_target, label_map).reshape(-1)

        with torch.no_grad():
            point = model(batch)
            feat = restore_point_features(point)

        feat = feat.reshape(-1, feat.shape[-1])
        logits = probe(feat).reshape(-1, NUM_CLASSES)

        if logits.shape[0] != target.numel():
            raise ValueError(
                f"logits and target size mismatch: logits={logits.shape}, target={target.shape}"
            )

        valid = target != IGNORE_INDEX
        if not valid.any():
            continue

        loss = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * valid.sum().item()
        total_pts += valid.sum().item()

    return total_loss / max(total_pts, 1)


@torch.no_grad()
def evaluate(model, probe, loader, device, label_map):
    model.eval()
    probe.eval()

    hist = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        raw_target = batch["segment"].long()
        target = remap_labels(raw_target, label_map).reshape(-1)

        point = model(batch)
        feat = restore_point_features(point)
        feat = feat.reshape(-1, feat.shape[-1])

        logits = probe(feat).reshape(-1, NUM_CLASSES)
        pred = logits.argmax(dim=-1)

        if pred.shape[0] != target.numel():
            raise ValueError(
                f"pred and target size mismatch: pred={pred.shape}, target={target.shape}"
            )

        valid = target != IGNORE_INDEX
        if not valid.any():
            continue

        target = target[valid]
        pred = pred[valid]

        inds = NUM_CLASSES * target + pred
        hist += torch.bincount(inds, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES).cpu()

    hist = hist.float()

    tp = torch.diag(hist)
    gt = hist.sum(dim=1)
    pred_count = hist.sum(dim=0)
    union = gt + pred_count - tp

    iou = tp / union.clamp(min=1)
    acc = tp / gt.clamp(min=1)

    valid_classes = gt > 0
    miou = iou[valid_classes].mean().item() if valid_classes.any() else 0.0
    macc = acc[valid_classes].mean().item() if valid_classes.any() else 0.0
    allacc = tp.sum().item() / gt.sum().item() if gt.sum() > 0 else 0.0

    return {
        "mIoU": miou,
        "mAcc": macc,
        "allAcc": allacc,
        "IoU_per_class": {CLASS_NAMES[i]: iou[i].item() for i in range(NUM_CLASSES)},
        "Acc_per_class": {CLASS_NAMES[i]: acc[i].item() for i in range(NUM_CLASSES)},
        "hist": hist,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest="data_path", help="path to LiDAR-Net data")
    parser.add_argument("--output_path", dest="output_path", help="path for checkpoint .pt file")
    parser.add_argument("--epochs", type=int, default=100, help="maximum training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="early stopping patience on validation mIoU",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_map = build_label_map().to(device)

    transform = utonia.transform.default(0.5)
    dataset = LiDARNetDataset(path=args.data_path, split="train", transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x[0],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],
    )

    model, probe = load_model(device, NUM_CLASSES)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = args.epochs
    best_miou = float("-inf")
    best_epoch = -1
    best_probe_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss = train(model, probe, train_loader, optimizer, device, label_map)
        metrics = evaluate(model, probe, val_loader, device, label_map)
        current_miou = metrics["mIoU"]

        if current_miou > best_miou:
            best_miou = current_miou
            best_epoch = epoch
            best_probe_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }
            epochs_without_improvement = 0
            early_stop_status = "improved"
        else:
            epochs_without_improvement += 1
            early_stop_status = f"no_improve={epochs_without_improvement}"

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"mIoU={metrics['mIoU']:.4f} "
            f"mAcc={metrics['mAcc']:.4f} "
            f"allAcc={metrics['allAcc']:.4f} "
            f"early_stop={early_stop_status}"
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch:03d}: "
                f"validation mIoU did not improve for {args.patience} epochs. "
                f"Best mIoU={best_miou:.4f} at epoch {best_epoch:03d}."
            )
            break

    if best_probe_state is None:
        best_probe_state = {
            key: value.detach().cpu().clone()
            for key, value in probe.state_dict().items()
        }

    os.makedirs(args.output_path, exist_ok=True)
    torch.save(
        {
            "probe": best_probe_state,
            "class_names": CLASS_NAMES,
            "raw_class_ids": RAW_CLASS_IDS,
            "best_val_mIoU": best_miou,
            "best_epoch": best_epoch,
        },
        os.path.join(args.output_path, "utonia_lidarnet.pt"),
    )
