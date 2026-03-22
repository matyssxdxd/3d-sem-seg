import trimesh
import numpy as np

def load_point_cloud(path: str) -> np.ndarray:
    mesh = trimesh.load(path, process=False)
    xyz = np.asarray(mesh.vertices, dtype=np.float32)

    return xyz 

if __name__ == "__main__":
    xyz = load_point_cloud("/home/matyss/Masters/scan_20_opt_denoised.ply")
    print(xyz.shape)
    print(xyz)
