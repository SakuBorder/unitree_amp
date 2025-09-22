"""
Add box pose/size fields into motion .pkl files.
KISS version: no warnings / verbose prints; fail fast on missing HTML deps.
"""

import argparse
import glob
import math
import os
import re
from typing import Dict, Tuple, Optional

import joblib
import numpy as np
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES

# Optional matplotlib preview (silent if unavailable)
try:
    import matplotlib.pyplot as plt  # type: ignore
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore
except Exception:
    plt = None
    Poly3DCollection = None

# Optional HTML export deps (validated strictly when --visualize-html is used)
try:
    from lpanlib.isaacgym_utils.vis.api import vis_hoi_use_scenepic_animation_climb  # type: ignore
except Exception:
    vis_hoi_use_scenepic_animation_climb = None

try:
    import trimesh  # type: ignore
except Exception:
    trimesh = None


DEFAULT_BACKWARD_OFFSET = 0.13
L_KNEE = SMPL_BONE_ORDER_NAMES.index("L_Knee")
R_KNEE = SMPL_BONE_ORDER_NAMES.index("R_Knee")

SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int32,
)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) -> 3x3 rotation."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / n, x / n, y / n, z / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _box_corners_local(size: np.ndarray) -> np.ndarray:
    L, W, H = size
    hx, hy, hz = L / 2.0, W / 2.0, H / 2.0
    return np.array(
        [
            [+hx, +hy, +hz],
            [+hx, +hy, -hz],
            [+hx, -hy, +hz],
            [+hx, -hy, -hz],
            [-hx, +hy, +hz],
            [-hx, +hy, -hz],
            [-hx, -hy, +hz],
            [-hx, -hy, -hz],
        ],
        dtype=np.float32,
    )


def _apply_pose_to_points(points_local: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    R = _quat_to_rotmat(quat_wxyz)
    return (points_local @ R.T) + pos[None, :]


def _parse_size_from_mesh_dir(mesh_dir: str) -> Optional[np.ndarray]:
    """Parse meters (L,W,H) from a dir name like .../box_length_2000_10_10 (mm)."""
    if not mesh_dir:
        return None
    base = os.path.basename(mesh_dir.rstrip("/"))
    m = re.search(r"(\d+)[^\d]+(\d+)[^\d]+(\d+)$", base)
    if not m:
        return None
    L_mm, W_mm, H_mm = map(float, m.groups())
    return np.array([L_mm, W_mm, H_mm], dtype=np.float32) / 1000.0


def _parse_size_from_arg(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError("--box-size must be 'L,W,H' in meters")
    return np.array(vals, dtype=np.float32)


def _compute_box_state(
    data: Dict[str, np.ndarray],
    backward_offset: float,
    box_size_m: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    """Return (pose7, knee_idx, frame) with pose7=[x,y,z,qw,qx,qy,qz] world."""
    joints = data["smpl_joints"]  # (N,24,3)
    root_rot = data["root_rot"]  # (N,4) (wxyz)

    z_left = joints[:, L_KNEE, 2]
    z_right = joints[:, R_KNEE, 2]
    l_frame = int(np.argmax(z_left))
    r_frame = int(np.argmax(z_right))
    use_left = z_left[l_frame] >= z_right[r_frame]
    frame = l_frame if use_left else r_frame
    knee_idx = L_KNEE if use_left else R_KNEE

    center = joints[frame, knee_idx].astype(np.float32).copy()
    center[2] = 0.0
    qw, qx, qy, qz = root_rot[frame].astype(np.float32)
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    # Keep only the yaw component so the obstacle remains level
    qw, qx, qy, qz = math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)
    facing = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
    center[:2] -= facing[:2] * float(backward_offset)

    L, W, H = map(float, box_size_m)
    corners_local = np.array(
        [
            [+L / 2, +W / 2, +H / 2],
            [+L / 2, +W / 2, -H / 2],
            [+L / 2, -W / 2, +H / 2],
            [+L / 2, -W / 2, -H / 2],
            [-L / 2, +W / 2, +H / 2],
            [-L / 2, +W / 2, -H / 2],
            [-L / 2, -W / 2, +H / 2],
            [-L / 2, -W / 2, -H / 2],
        ],
        dtype=np.float32,
    )
    R = _quat_to_rotmat(np.array([qw, qx, qy, qz], dtype=np.float32))
    corners_world = corners_local @ R.T + center[None, :]
    min_z = float(np.min(corners_world[:, 2]))
    if min_z < 0:
        center[2] -= min_z  # lift so the lowest corner touches ground

    pose7 = np.array([center[0], center[1], center[2], qw, qx, qy, qz], dtype=np.float32)
    return pose7, knee_idx, frame


def _visualize(data: Dict[str, np.ndarray], frame: int, corners_world: np.ndarray) -> None:
    if plt is None or Poly3DCollection is None:
        return
    joints = data["smpl_joints"][frame]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i, p in enumerate(SMPL_PARENTS):
        if p >= 0:
            ax.plot([joints[p, 0], joints[i, 0]], [joints[p, 1], joints[i, 1]], [joints[p, 2], joints[i, 2]])

    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=10)

    faces = [
        [corners_world[i] for i in [0, 1, 3, 2]],
        [corners_world[i] for i in [4, 5, 7, 6]],
        [corners_world[i] for i in [0, 1, 5, 4]],
        [corners_world[i] for i in [2, 3, 7, 6]],
        [corners_world[i] for i in [0, 2, 6, 4]],
        [corners_world[i] for i in [1, 3, 7, 5]],
    ]
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.3))

    all_pts = np.vstack([joints, corners_world])
    max_range = np.ptp(all_pts, axis=0).max() / 2.0
    mid = all_pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    plt.show()


def _export_html(
    data: Dict[str, np.ndarray],
    box_state: np.ndarray,
    box_size_m: np.ndarray,
    out_html: str,
    humanoid_xml: str,
) -> None:
    if vis_hoi_use_scenepic_animation_climb is None:
        raise ImportError("`--visualize-html` requires lpanlib.")
    if trimesh is None:
        raise ImportError("`--visualize-html` requires trimesh.")
    joints = data.get("smpl_joints")
    if joints is None:
        raise ValueError("Missing `smpl_joints` in data.")
    rots = data.get("smpl_joint_quats")
    if rots is None:
        rots = np.zeros((joints.shape[0], joints.shape[1], 4), dtype=np.float32)
        rots[..., 3] = 1.0  # identity (xyzw)

    mesh = trimesh.creation.box(extents=box_size_m)
    obj_global_pos = [box_state[:3]]
    obj_global_rot = [np.array([box_state[4], box_state[5], box_state[6], box_state[3]], dtype=np.float32)]
    obj_colors = [(255, 255, 0)]
    obj_names = [data.get("box_name", "Box")]

    vis_hoi_use_scenepic_animation_climb(
        asset_filename=humanoid_xml,
        rigidbody_global_pos=joints,
        rigidbody_global_rot=rots,
        fps=float(data.get("mocap_framerate", 30)),
        up_axis="z",
        color=(240, 248, 255),
        output_path=out_html,
        obj_meshes=[mesh],
        obj_global_pos=obj_global_pos,
        obj_global_rot=obj_global_rot,
        obj_colors=obj_colors,
        obj_names=obj_names,
    )


def process_file(
    path: str,
    suffix: str,
    box_size_m: np.ndarray,
    box_name: str,
    write_corners: bool,
    backward_offset: float,
    mesh_dir: Optional[str],
    visualize: bool,
    visualize_html: bool,
    humanoid_xml: Optional[str],
) -> str:
    obj = joblib.load(path)
    if len(obj) != 1:
        raise ValueError(f"Unexpected pkl structure in {path}")
    key, data = next(iter(obj.items()))
    data = dict(data)

    box_state, knee_idx, frame = _compute_box_state(data, backward_offset, box_size_m)

    data["box_state"] = box_state  # [x y z qw qx qy qz]
    data["box_size"] = box_size_m.astype(np.float32)  # [L W H]
    data["box_name"] = box_name

    corners_world = None
    if write_corners or visualize or visualize_html:
        pos = box_state[:3]
        quat = box_state[3:]
        corners_local = _box_corners_local(box_size_m)
        corners_world = _apply_pose_to_points(corners_local, pos, quat)
        if write_corners:
            data["box_corners_world"] = corners_world.astype(np.float32)

    if visualize and corners_world is not None:
        _visualize(data, frame, corners_world)

    if visualize_html:
        if not humanoid_xml:
            raise ValueError("`--visualize-html` requires `--humanoid-xml PATH`.")
        _export_html(
            data=data,
            box_state=box_state,
            box_size_m=box_size_m,
            out_html=path.replace(".pkl", f"{suffix}.html"),
            humanoid_xml=humanoid_xml,
        )

    provenance = {
        "used_knee": "L_Knee" if knee_idx == L_KNEE else "R_Knee",
        "backward_offset_m": float(backward_offset),
    }
    if mesh_dir:
        provenance["mesh_dir"] = mesh_dir
    data["box_meta"] = provenance

    out_path = path.replace(".pkl", f"{suffix}.pkl")
    joblib.dump({key: data}, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pkl_dir", help="Directory containing motion .pkl files")
    parser.add_argument("--suffix", default="_box", help="Suffix appended to output files (before .pkl)")
    parser.add_argument("--box-name", default="Box", help="Name tag for the box")
    parser.add_argument(
        "--box-size",
        default=None,
        help="Box size in meters as 'L,W,H' (overrides mesh-dir parsing).",
    )
    parser.add_argument(
        "--mesh-dir",
        default=None,
        help="Dir like '/.../box_length_2000_10_10' to parse size (mm->m).",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=DEFAULT_BACKWARD_OFFSET,
        help=f"Backward offset in meters (default {DEFAULT_BACKWARD_OFFSET})",
    )
    parser.add_argument(
        "--write-corners",
        action="store_true",
        help="Also write box_corners_world (8x3).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Matplotlib 3D preview (silent if matplotlib not installed).",
    )
    parser.add_argument(
        "--visualize-html",
        action="store_true",
        help="Export interactive HTML via ScenePic (requires lpanlib & trimesh).",
    )
    parser.add_argument(
        "--humanoid-xml",
        default=None,
        help="Humanoid XML asset file for --visualize-html.",
    )
    args = parser.parse_args()

    # Hard-check HTML visualization dependencies up front (fail fast).
    if args.visualize_html:
        if vis_hoi_use_scenepic_animation_climb is None:
            raise ImportError("`--visualize-html` requires `lpanlib`.")
        if trimesh is None:
            raise ImportError("`--visualize-html` requires `trimesh`.")
        if not args.humanoid_xml:
            raise ValueError("`--visualize-html` requires `--humanoid-xml PATH`.")
        if not os.path.isfile(args.humanoid_xml):
            raise FileNotFoundError(f"Humanoid XML not found: {args.humanoid_xml}")

    # Resolve size (meters)
    size_from_arg = _parse_size_from_arg(args.box_size) if args.box_size else None
    size_from_mesh = _parse_size_from_mesh_dir(args.mesh_dir) if args.mesh_dir else None
    if size_from_arg is not None:
        box_size_m = size_from_arg
    elif size_from_mesh is not None:
        box_size_m = size_from_mesh
    else:
        box_size_m = np.array([0.5, 0.3, 0.3], dtype=np.float32)

    files = sorted(glob.glob(os.path.join(args.pkl_dir, "*.pkl")))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {args.pkl_dir}")

    for f in files:
        process_file(
            path=f,
            suffix=args.suffix,
            box_size_m=box_size_m,
            box_name=args.box_name,
            write_corners=args.write_corners,
            backward_offset=args.offset,
            mesh_dir=args.mesh_dir,
            visualize=args.visualize,
            visualize_html=args.visualize_html,
            humanoid_xml=args.humanoid_xml,
        )


if __name__ == "__main__":
    main()