import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import utonia
from plyfile import PlyData, PlyElement

try:
    import flash_attn
    print("Using flash attention for faster inference.")
except ImportError:
    flash_attn = None


# 0 is unannotated / unknown and is not part of the 24 annotated classes.
RAW_CLASS_INFO = [
    ("Unknown", 0),
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

RAW_LABEL_TO_NAME = {raw_id: name for name, raw_id in RAW_CLASS_INFO}
DEFAULT_RAW_CLASS_IDS = [raw_id for name, raw_id in RAW_CLASS_INFO if raw_id != 0]

RAW_COLORMAP = {
    0:  (0.0, 0.0, 0.0),        # Unknown - black
    3:  (0.2, 0.8, 0.2),        # Houseplant - green
    4:  (0.1, 0.5, 0.1),        # Tree - darker green
    5:  (1.0, 0.0, 0.0),        # Person - bright red
    15: (0.0, 0.7, 1.0),        # Window - bright cyan
    17: (1.0, 0.5, 0.0),        # Door - bright orange
    9:  (0.55, 0.55, 0.60),     # Floor
    11: (0.75, 0.75, 0.80),     # Ceiling (lighter)
    13: (0.65, 0.65, 0.70),     # Wall
    10: (0.50, 0.60, 0.70),     # Stair (slightly bluish)
    14: (0.45, 0.55, 0.65),     # Pillar
    12: (0.6, 0.4, 0.2),        # Pipe - brown
    16: (0.6, 0.6, 0.9),        # Curtain - soft blue
    18: (0.6, 0.3, 0.7),        # Table
    19: (0.7, 0.5, 0.4),        # Chair
    20: (0.9, 0.6, 0.6),        # Sofa
    21: (0.4, 0.0, 0.4),        # Blackboard
    22: (0.5, 0.0, 0.7),        # Monitor
    23: (0.9, 0.9, 0.3),        # Bookshelf
    24: (0.7, 0.7, 0.4),        # Wardrobe
    25: (0.4, 0.4, 0.4),        # Bed
    26: (0.0, 0.5, 1.0),        # Reflection noise - blue
    27: (1.0, 0.2, 0.2),        # Ghost - reddish
    29: (1.0, 1.0, 0.6),        # Light - bright yellow
    30: (0.7, 0.7, 0.7),        # Tabletop others - neutral
}

RAW_LUT = np.zeros((max(RAW_COLORMAP) + 1, 3), dtype=np.uint8)
for raw_id, color in RAW_COLORMAP.items():
    RAW_LUT[raw_id] = np.round(255.0 * np.array(color, dtype=np.float32)).astype(np.uint8)


def load_point_cloud(path):
    """
    Load XYZ and RGB from a PLY point cloud.

    The LiDAR-Net rooms use PLY files with at least x, y, z, red, green, blue.
    If RGB is absent, zeros are used.
    """
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"PLY file has no vertex element: {path}")

    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or [])
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise ValueError(f"PLY vertex element must contain x/y/z fields: {path}")

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32)
    else:
        rgb = np.zeros_like(xyz, dtype=np.float32)

    return xyz, rgb


def restore_point_output(point):
    """
    Undo Utonia pooling hierarchy and recover per-point features and coordinates
    in the same order.
    """
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent

    feat = point.feat
    coord = point.coord
    if "inverse" in point.keys():
        feat = feat[point.inverse]
        coord = coord[point.inverse]
    return feat, coord


def load_model(device, num_classes, transform):
    if flash_attn is not None:
        model = utonia.load("utonia", repo_id="Pointcept/Utonia").to(device)
    else:
        custom_config = {
            "enc_patch_size": [1024 for _ in range(5)],
            "enable_flash": False,
        }
        model = utonia.load(
            "utonia",
            repo_id="Pointcept/Utonia",
            custom_config=custom_config,
        ).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dummy = {
        "coord": np.random.randn(4096, 3).astype(np.float32),
        "color": np.random.randint(0, 256, size=(4096, 3)).astype(np.float32),
        "normal": np.zeros((4096, 3), dtype=np.float32),
        "segment": np.zeros(4096, dtype=np.int32),
    }
    dummy = transform(dummy)

    with torch.inference_mode():
        for k, v in dummy.items():
            if isinstance(v, torch.Tensor):
                dummy[k] = v.to(device)
        point = model(dummy)
        feat, _ = restore_point_output(point)
        in_dim = feat.shape[-1]

    probe = nn.Linear(in_dim, num_classes).to(device)
    return model, probe


def save_colored_point_cloud(path, coords, colors_uint8, sem_labels=None, label_map=None):
    """
    Save colored point cloud to binary PLY with optional semantic ids.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    coords = np.asarray(coords, dtype=np.float32)
    colors_uint8 = np.asarray(colors_uint8, dtype=np.uint8)

    if coords.shape[0] != colors_uint8.shape[0]:
        raise ValueError(f"coords/colors size mismatch: {coords.shape} vs {colors_uint8.shape}")

    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    if sem_labels is not None:
        sem_labels = np.asarray(sem_labels, dtype=np.uint16)
        if sem_labels.shape[0] != coords.shape[0]:
            raise ValueError(f"coords/labels size mismatch: {coords.shape} vs {sem_labels.shape}")
        vertex_dtype.append(("sem", "u2"))

    vertex = np.empty(coords.shape[0], dtype=vertex_dtype)
    vertex["x"] = coords[:, 0]
    vertex["y"] = coords[:, 1]
    vertex["z"] = coords[:, 2]
    vertex["red"] = colors_uint8[:, 0]
    vertex["green"] = colors_uint8[:, 1]
    vertex["blue"] = colors_uint8[:, 2]
    if sem_labels is not None:
        vertex["sem"] = sem_labels

    comments = []
    if label_map is not None:
        comments = [f"sem_{idx}={name}" for idx, name in sorted(label_map.items())]

    ply = PlyData([PlyElement.describe(vertex, "vertex")], text=False, comments=comments)
    ply.write(path)


def load_probe_checkpoint(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "probe" in ckpt:
        state_dict = ckpt["probe"]
        class_names = ckpt.get("class_names")
        raw_class_ids = ckpt.get("raw_class_ids")
        return state_dict, class_names, raw_class_ids

    return ckpt, None, None


def get_num_classes_from_state_dict(state_dict):
    if "weight" not in state_dict:
        raise KeyError("Probe state_dict must contain a 'weight' tensor.")
    return int(state_dict["weight"].shape[0])


def build_output_mapping(num_classes, raw_class_ids=None, class_names=None):
    """
    Infer how probe outputs map to raw semantic ids.

    Supported cases:
      - corrected checkpoint: 24 logits over contiguous train ids, mapped with raw_class_ids
      - legacy checkpoint: 31 logits directly over raw semantic ids 0..30
    """
    if raw_class_ids is not None:
        raw_class_ids = [int(x) for x in raw_class_ids]
        if len(raw_class_ids) != num_classes:
            raise ValueError(
                f"Checkpoint raw_class_ids has length {len(raw_class_ids)} but probe outputs {num_classes} classes."
            )
        output_to_raw = np.asarray(raw_class_ids, dtype=np.int64)
    elif num_classes == len(DEFAULT_RAW_CLASS_IDS):
        output_to_raw = np.asarray(DEFAULT_RAW_CLASS_IDS, dtype=np.int64)
    elif num_classes == 31:
        output_to_raw = np.arange(31, dtype=np.int64)
    else:
        raise ValueError(
            "Could not infer label mapping from checkpoint. "
            f"Probe outputs {num_classes} classes; expected 24 (remapped LiDAR-Net) or 31 (legacy raw ids)."
        )

    if class_names is not None and len(class_names) == len(output_to_raw):
        output_label_map = {int(i): str(name) for i, name in enumerate(class_names)}
    else:
        output_label_map = {int(i): RAW_LABEL_TO_NAME.get(int(raw_id), f"class_{int(raw_id)}") for i, raw_id in enumerate(output_to_raw)}

    raw_label_map = {int(raw_id): RAW_LABEL_TO_NAME.get(int(raw_id), f"class_{int(raw_id)}") for raw_id in np.unique(output_to_raw)}
    if 0 not in raw_label_map:
        raw_label_map[0] = RAW_LABEL_TO_NAME.get(0, "Unknown")

    return output_to_raw, output_label_map, raw_label_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", dest="data_path", required=True, help="path to .ply input data")
    parser.add_argument("--output_path", dest="output_path", required=True, help="path for the output .ply file")
    parser.add_argument("--ckpt_path", dest="ckpt_path", required=True, help="path to the ckpt for the linear probe")
    parser.add_argument(
        "--max_points",
        type=int,
        default=1_000_000,
        help="optionally subsample to at most this many points before inference",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed used for subsampling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = utonia.transform.default(0.5)

    probe_state_dict, ckpt_class_names, ckpt_raw_class_ids = load_probe_checkpoint(args.ckpt_path)
    num_classes = get_num_classes_from_state_dict(probe_state_dict)
    output_to_raw, _, raw_label_map = build_output_mapping(
        num_classes,
        raw_class_ids=ckpt_raw_class_ids,
        class_names=ckpt_class_names,
    )

    model, probe = load_model(device, num_classes, transform)
    probe.load_state_dict(probe_state_dict)
    model.eval()
    probe.eval()

    xyz, rgb = load_point_cloud(args.data_path)

    if args.max_points is not None and xyz.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(xyz.shape[0], args.max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    point = {
        "coord": xyz.astype(np.float32),
        "color": rgb.astype(np.float32),
        "normal": np.zeros_like(xyz, dtype=np.float32),
        "segment": np.zeros(len(xyz), dtype=np.int32),  # dummy labels for transform compatibility
    }

    print("Applying transforms...")
    point = transform(point)

    print("Running inference...")
    with torch.inference_mode():
        for key, value in point.items():
            if isinstance(value, torch.Tensor):
                point[key] = value.to(device, non_blocking=(device.type == "cuda"))

        point = model(point)
        feat, coords = restore_point_output(point)
        seg_logits = probe(feat)
        pred_out = seg_logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)

    pred_raw = output_to_raw[pred_out].astype(np.uint16)
    pred_color = RAW_LUT[pred_raw]
    coords = coords.detach().cpu().numpy().astype(np.float32)

    save_colored_point_cloud(
        args.output_path,
        coords,
        pred_color,
        sem_labels=pred_raw,
        label_map=raw_label_map,
    )

    print(f"Saved semantic segmentation visualization to: {args.output_path}")
    unique_raw = np.unique(pred_raw)
    unique_names = [RAW_LABEL_TO_NAME.get(int(raw_id), f"class_{int(raw_id)}") for raw_id in unique_raw]
    print("Predicted semantic ids:", unique_raw.tolist())
    print("Predicted semantic names:", unique_names)


if __name__ == "__main__":
    main()
