import argparse
import utonia
import torch
import torch.nn as nn
import numpy as np
import trimesh
import os
from plyfile import PlyData, PlyElement

try:
    import flash_attn
    print("Using flash attention for faster inference.")
except ImportError:
    flash_attn = None

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

LIDAR_NET_LUT = (255.0 * np.array([LIDAR_NET_COLORMAP[i] for i in range(31)], dtype=np.float32)).astype(np.uint8)
LIDAR_NET_CLASS_LABELS = (
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
    "tabletop others",
)
LIDAR_NET_LABELS = {
    i: (LIDAR_NET_CLASS_LABELS[i] if i < len(LIDAR_NET_CLASS_LABELS) else f"class_{i}")
    for i in range(31)
}


def load_point_cloud_xyz(path):
    """
    Load XYZ coordinates from a PLY/point cloud file using trimesh.
    """
    mesh = trimesh.load(path, process=False)

    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        if not geoms:
            raise ValueError(f"No geometry found in file: {path}")

        collected = []
        for geom in geoms:
            if hasattr(geom, "vertices"):
                verts = np.asarray(geom.vertices, dtype=np.float32)
                if verts.size > 0:
                    collected.append(verts)
            elif hasattr(geom, "points"):
                pts = np.asarray(geom.points, dtype=np.float32)
                if pts.size > 0:
                    collected.append(pts)

        if not collected:
            raise ValueError(f"Could not extract points from scene: {path}")

        xyz = np.concatenate(collected, axis=0)
        return xyz

    if hasattr(mesh, "vertices"):
        xyz = np.asarray(mesh.vertices, dtype=np.float32)
        if xyz.size == 0:
            raise ValueError(f"No vertices found in file: {path}")
        return xyz

    if hasattr(mesh, "points"):
        xyz = np.asarray(mesh.points, dtype=np.float32)
        if xyz.size == 0:
            raise ValueError(f"No points found in file: {path}")
        return xyz

    raise ValueError(f"Unsupported geometry type in file: {path}")

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

def save_colored_point_cloud(path, coords, colors_uint8, labels=None, label_map=None):
    """
    Save colored point cloud to PLY with optional label ids and metadata.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    coords = np.asarray(coords, dtype=np.float32)
    colors_uint8 = np.asarray(colors_uint8, dtype=np.uint8)

    if labels is not None:
        labels = np.asarray(labels, dtype=np.uint16)

    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    if labels is not None:
        vertex_dtype.append(("label", "u2"))

    vertex = np.empty(coords.shape[0], dtype=vertex_dtype)
    vertex["x"] = coords[:, 0]
    vertex["y"] = coords[:, 1]
    vertex["z"] = coords[:, 2]
    vertex["red"] = colors_uint8[:, 0]
    vertex["green"] = colors_uint8[:, 1]
    vertex["blue"] = colors_uint8[:, 2]
    if labels is not None:
        vertex["label"] = labels

    comments = []
    if label_map is not None:
        comments = [f"label_{idx}={name}" for idx, name in sorted(label_map.items())]

    ply = PlyData([PlyElement.describe(vertex, "vertex")], text=False, comments=comments)
    ply.write(path)


def load_probe_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "probe" in ckpt:
        return ckpt["probe"]
    return ckpt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", dest="data_path", help="path to .ply input data"
    )
    parser.add_argument(
        "--output_path", dest="output_path", help="path for the output .ply file"
    )
    parser.add_argument(
        "--ckpt_path", dest="ckpt_path", help="path to the ckpt for the linear probe"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = args.data_path 
    num_classes = 31

    transform = utonia.transform.default(0.5)

    model, probe = load_model(device, num_classes)
    probe.load_state_dict(load_probe_state_dict(args.ckpt_path))

    model.eval()
    probe.eval()

    # Load default data transform pipeline
    transform = utonia.transform.default(0.5)

    # Load input point cloud
    xyz = load_point_cloud_xyz(args.data_path)

    npoints = 1000000
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.shape[0] > npoints:
        idx = np.random.choice(xyz.shape[0], npoints, replace=False)
        xyz = xyz[idx]

    normals = np.zeros_like(xyz, dtype=np.float32)
    colors = np.zeros_like(xyz, dtype=np.float32)

    point = {
        "coord": xyz.astype(np.float32),
        "color": colors.astype(np.float32),
        "normal": normals.astype(np.float32),
        "segment": np.zeros(len(xyz), dtype=np.int32),  # dummy labels
    }

    print("Applying transforms...")
    point = transform(point)

    # Inference
    print("Running inference...")

    model.eval()
    probe.eval()
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor) and device == "cuda":
                point[key] = point[key].cuda(non_blocking=True)

        point = model(point)

        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        feat = point.feat
        seg_logits = probe(feat)
        pred = seg_logits.argmax(dim=-1).data.cpu().numpy()
        color = LIDAR_NET_LUT[pred]

    # Save result
    coords = point.coord.cpu().detach().numpy()
    save_colored_point_cloud(
        args.output_path,
        coords,
        color,
        labels=pred,
        label_map=LIDAR_NET_LABELS,
    )

    print(f"Saved semantic segmentation visualization to: {args.output_path}")
