from collections import OrderedDict
import numpy as np
import torch
import trimesh
import argparse

from huggingface_hub import hf_hub_download

from datasets.transform import Compose
from models.default import DefaultSegmentorV2

def load_point_cloud(path: str):
    mesh = trimesh.load(path, process=False)
    xyz = np.asarray(mesh.vertices, dtype=np.float32)

    return xyz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
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

    model.load_state_dict(ckpt["state_dict"], strict=False)
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

    print("Prediction shape:", pred.shape)
    print("Unique labels:", np.unique(pred))
