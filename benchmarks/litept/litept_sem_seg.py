from collections import OrderedDict
import numpy as np
import torch
import trimesh
import argparse
import os

from huggingface_hub import hf_hub_download

from datasets.transform import Compose
from models.default import DefaultSegmentorV2

VALID_CLASS_IDS_20 = (
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
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)

CLASS_LABELS_20 = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

CLASS_COLOR_20 = [SCANNET_COLOR_MAP_20[class_id] for class_id in VALID_CLASS_IDS_20]

def load_point_cloud(path: str):
    mesh = trimesh.load(path, process=False)
    xyz = np.asarray(mesh.vertices, dtype=np.float32)

    return xyz

def save_colored_point_cloud(path, coords, colors_uint8):
    """
    Save colored point cloud to PLY using trimesh.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cloud = trimesh.points.PointCloud(
        vertices=np.asarray(coords, dtype=np.float32),
        colors=np.asarray(colors_uint8, dtype=np.uint8),
    )
    cloud.export(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument(
        "--output_file", dest="output_file", help="where to output .ply with results"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DefaultSegmentorV2(
        num_classes=20,
        backbone_out_channels=72,
        backbone=dict(
            type="LitePT",
            in_channels=6,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(36, 72, 144, 252, 504),
            enc_num_head=(2, 4, 8, 14, 28),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            enc_conv=(True, True, True, False, False),
            enc_attn=(False, False, False, True, True),
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=0.3,
            shuffle_orders=True,
            pre_norm=True,
        ),
    )

    model = model.to(device)

    ckpt_path = hf_hub_download(
        repo_id="prs-eth/LitePT",
        filename="scannet-semseg-litept-small-v1m1/model/model_best.pth",
        repo_type="model"
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")

    print("RUNNING SCRIPT:", __file__)
    print("CKPT PATH:", ckpt_path)
    raw = ckpt["state_dict"]
    print("ckpt key sample:", list(raw.keys())[:3])          # should show module.*
    print("model key sample:", list(model.state_dict().keys())[:3])  # no module.*

    state = ckpt["state_dict"]

    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}

    mk = set(model.state_dict().keys())
    ck = set(state.keys())
    print("overlap:", len(mk & ck), "model:", len(mk), "ckpt:", len(ck))
    info = model.load_state_dict(state, strict=False)
    print("missing:", len(info.missing_keys))
    print("unexpected:", len(info.unexpected_keys))

    model.eval()

    xyz = load_point_cloud(args.input_path)

    rgb = np.zeros_like(xyz, dtype=np.float32)

    normals = np.zeros_like(xyz, dtype=np.float32)

    point = {
        "coord": xyz.astype(np.float32),
        "color": rgb.astype(np.float32),
        "normal": normals.astype(np.float32),
    }

    print("Points:", xyz.shape)

    data_config = [
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="CenterShift", apply_z=False),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "inverse"),
            feat_keys=("color", "normal"),
        ),
    ]

    transform = Compose(data_config)
    point = transform(point)

    with torch.no_grad():
        for k in point.keys():
            if isinstance(point[k], torch.Tensor):
                point[k] = point[k].to(device, non_blocking=True)

        output = model(point)

        logits = output["seg_logits"]            # [N_down, num_classes]

        dense_logits = logits[point["inverse"]]  # [N, num_classes]
        pred = dense_logits.argmax(dim=1).cpu().numpy()
        color = np.array(CLASS_COLOR_20, dtype=np.uint8)[pred]

    color = np.array(CLASS_COLOR_20, dtype=np.uint8)[pred]

    print("Prediction shape:", pred.shape)
    print("Unique labels:", np.unique(pred))

    assert xyz.shape[0] == color.shape[0], (
        f"Point/color count mismatch: xyz={xyz.shape[0]}, color={color.shape[0]}"
    )

    save_colored_point_cloud(args.output_file, xyz, color)

    print(f"Saved semantic segmentation visualization to: {args.output_file}")
