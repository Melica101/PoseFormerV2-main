import re
from pathlib import Path
import numpy as np
import scipy.io

RES_W, RES_H = 346, 260

P_LIST = [
    np.load("DHP19/P_matrices/P1.npy"),
    np.load("DHP19/P_matrices/P2.npy"),
    np.load("DHP19/P_matrices/P3.npy"),
    np.load("DHP19/P_matrices/P4.npy"),
]
assert all(P.shape == (3, 4) for P in P_LIST), "Each P must be 3x4"

ORDER_13 = (
    "hipL",       # 0 root
    "head",       # 1
    "shoulderR",  # 2
    "shoulderL",  # 3
    "elbowR",     # 4
    "elbowL",     # 5
    "hipR",       # 6
    "handR",      # 7
    "handL",      # 8
    "kneeR",      # 9
    "kneeL",      # 10
    "footR",      # 11
    "footL",      # 12
)

NUM_JOINTS = 13
KPS_LEFT  = [3, 5, 8, 10, 12]
KPS_RIGHT = [2, 4, 7, 6, 9, 11]

def parse_name(mat_path: Path):
    # S10_1_1.mat -> subject=S10 action="1_1"
    m = re.match(r"^(S\d+)_([0-9]+)_([0-9]+)\.mat$", mat_path.name)
    if not m:
        raise ValueError(f"Unexpected filename: {mat_path.name} (expected S<subj>_<sess>_<mov>.mat)")
    subj, sess, mov = m.group(1), m.group(2), m.group(3)
    return subj, f"{sess}_{mov}"

def load_xyzpos_as_X13(mat_path: Path) -> np.ndarray:
    m = scipy.io.loadmat(str(mat_path))
    if "XYZPOS" not in m:
        raise KeyError(f"{mat_path} missing XYZPOS")

    xyz = m["XYZPOS"][0, 0]  # structured scalar
    # Build X in ORDER_13
    joints = []
    for name in ORDER_13:
        arr = xyz[name]
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{mat_path}: XYZPOS[{name}] shape {arr.shape}, expected (F,3)")
        joints.append(arr.astype(np.float32))
    X13 = np.stack(joints, axis=1)  # (F,13,3)
    return X13

def make_relative_keep_root0(X: np.ndarray) -> np.ndarray:
    # mimic H36M: keep joint 0 as trajectory/root, make others relative
    X = X.copy()
    X[:, 1:, :] -= X[:, :1, :]
    return X

def project_points(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    F, J, _ = X.shape
    Xh = np.concatenate([X, np.ones((F, J, 1), dtype=X.dtype)], axis=-1)  # (F,J,4)
    xh = Xh @ P.T  # (F,J,3)
    uv = xh[..., :2] / xh[..., 2:3]
    return uv.astype(np.float32)

def main(vicon_dir: str, out_dir: str = "data"):
    vicon_dir = Path(vicon_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positions_3d = {}  # subject -> action -> dict('positions_3d': [cam0..cam3])
    positions_2d = {}  # subject -> action -> [cam0..cam3]
    cameras = {}       # subject -> [cam dicts]

    mat_files = sorted(vicon_dir.glob("S*_*.mat"))
    if not mat_files:
        raise RuntimeError(f"No Vicon .mat files found in {vicon_dir}")

    for mat_path in mat_files:
        subj, action = parse_name(mat_path)

        X13 = load_xyzpos_as_X13(mat_path)          # (F,13,3)
        X13 = make_relative_keep_root0(X13)         # (F,13,3), root=hipL at joint 0

        # 3D list-per-camera (same GT repeated; PoseFormer code expects list)
        X_list = [X13 for _ in range(4)]

        # 2D GT per camera via P projection (pixel coords)
        U_list = [project_points(P_LIST[i], X13) for i in range(4)]

        positions_3d.setdefault(subj, {})
        positions_2d.setdefault(subj, {})

        positions_3d[subj][action] = {"positions_3d": X_list}
        positions_2d[subj][action] = U_list

        if subj not in cameras:
            cameras[subj] = [{"res_w": RES_W, "res_h": RES_H} for _ in range(4)]

        print(f"OK {mat_path.name}: {X13.shape}")

    np.savez_compressed(out_dir / "data_3d_dhp19.npz",
                        positions_3d=positions_3d,
                        cameras=cameras)

    metadata = {
        "num_joints": NUM_JOINTS,
        "keypoints_symmetry": (KPS_LEFT, KPS_RIGHT),
        "layout_name": "dhp19_13j_hipL_root",
        "joint_names": list(ORDER_13),
    }
    np.savez_compressed(out_dir / "data_2d_dhp19_gt.npz",
                        positions_2d=positions_2d,
                        metadata=metadata)

    print("\nWrote:")
    print(" -", out_dir / "data_3d_dhp19.npz")
    print(" -", out_dir / "data_2d_dhp19_gt.npz")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vicon_dir", required=True)
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()
    main(args.vicon_dir, args.out_dir)
