import os
import sys
import time
import argparse
import os.path as osp
import glob

sys.path.append(os.getcwd())

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
import torch

import numpy as np
import math
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
import hydra
from omegaconf import DictConfig

# ---------------------------
# 原有的辅助：画胶囊点
# ---------------------------
def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                             mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                             float(point1[0]), float(point1[1]), float(point1[2]),
                             float(point2[0]), float(point2[1]), float(point2[2]))

# ---------------------------
# 新增：四元数/矩阵/box 可视化工具
# ---------------------------
def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """SciPy as_quat() 通常给 xyzw，这里转 wxyz。"""
    if q_xyzw.shape[-1] != 4:
        return q_xyzw
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

def _quat_to_rotmat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    w, x, y, z = map(float, q_wxyz)
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w/n, x/n, y/n, z/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ], dtype=np.float32)

def _set_geom_mat(g, R: np.ndarray):
    """兼容设置 mjvGeom.mat：R 为 (3,3)；g.mat 可能是 (9,) 或 (3,3)。"""
    M = R.astype(np.float32)
    try:
        if hasattr(g, "mat"):
            if isinstance(g.mat, np.ndarray):
                if g.mat.shape == (9,):
                    g.mat[:] = M.ravel()
                elif g.mat.shape == (3, 3):
                    g.mat[:, :] = M
                else:
                    flat = g.mat.reshape(-1)
                    flat[:min(9, flat.size)] = M.ravel()[:min(9, flat.size)]
            else:
                try:
                    g.mat[:] = M.ravel()
                except Exception:
                    pass
    except Exception:
        pass

def add_visual_box(scene, center_xyz, size_LWH, quat_wxyz, rgba):
    """在 user_scn 里添加一个 mjGEOM_BOX，返回其索引。注意：size 传半长。"""
    if scene.ngeom >= scene.maxgeom:
        return None
    scene.ngeom += 1
    idx = scene.ngeom - 1
    size_half = np.asarray(size_LWH, dtype=np.float32) / 2.0
    pos = np.asarray(center_xyz, dtype=np.float32)
    R = _quat_to_rotmat_wxyz(quat_wxyz)
    mujoco.mjv_initGeom(scene.geoms[idx],
                        mujoco.mjtGeom.mjGEOM_BOX,
                        size_half, pos, R.astype(np.float32).ravel(),
                        np.asarray(rgba, dtype=np.float32))
    _set_geom_mat(scene.geoms[idx], R)
    return idx

def update_visual_box(scene, idx, center_xyz, size_LWH, quat_wxyz):
    """更新已存在的 box 的位姿与尺寸。"""
    g = scene.geoms[idx]
    g.size[:] = (np.asarray(size_LWH, dtype=np.float32) / 2.0)
    g.pos[:]  = np.asarray(center_xyz, dtype=np.float32)
    R = _quat_to_rotmat_wxyz(quat_wxyz)
    _set_geom_mat(g, R)

# ---------------------------
# KISS：关节检查工具
# ---------------------------
def build_hinge_joint_table(mj_model):
    """
    收集所有 1-DoF hinge 关节（跳过 free root 的 7 个 qpos），
    返回 [{'name': str, 'qpos_adr': int}] 与 name->idx。
    """
    hinge = []
    name2idx = {}
    for j in range(mj_model.njnt):
        if mj_model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        adr = mj_model.jnt_qposadr[j]
        nm  = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        if adr < 7:
            continue
        hinge.append({'name': nm, 'qpos_adr': adr})
        name2idx[nm] = len(hinge) - 1
    return hinge, name2idx

def dump_current_angles(mj_model, mj_data, hinge_table, only_legs=True):
    """
    打印当前帧各关节角度（度）。默认仅腿部（HIP/KNEE/ANKLE）。
    """
    rows = []
    for rec in hinge_table:
        nm, adr = rec['name'], rec['qpos_adr']
        if only_legs and not any(k in nm.upper() for k in ["HIP", "KNEE", "ANKLE"]):
            continue
        ang = float(mj_data.qpos[adr]) * 180.0 / math.pi
        rows.append((nm, ang))
    rows.sort(key=lambda x: x[0])
    print("\n[JOINT ANGLES @ current frame] (deg)")
    for nm, ang in rows:
        print(f"{nm:16s} : {ang:+8.2f}")
    print("-" * 40)

def asymmetry_report(buffer_angles, pairs):
    """
    对时间窗 buffer_angles（list of dict(name->rad)）做左右对称报告。
    输出每对 (L_name, R_name) 的平均角度(度)与 |L|-|R|。
    """
    if not buffer_angles:
        return
    avg = {}
    for snap in buffer_angles:
        for nm, v in snap.items():
            avg.setdefault(nm, 0.0)
            avg[nm] += v
    for nm in avg.keys():
        avg[nm] /= len(buffer_angles)

    print("\n[ASYMMETRY REPORT] mean angles (deg) & |L|-|R| (deg)")
    offenders = []
    for ln, rn in pairs:
        if ln not in avg or rn not in avg:
            continue
        ldeg = avg[ln] * 180.0 / math.pi
        rdeg = avg[rn] * 180.0 / math.pi
        diff = abs(ldeg) - abs(rdeg)
        offenders.append((abs(diff), ln, rn, ldeg, rdeg, diff))
    offenders.sort(reverse=True)
    for score, ln, rn, ldeg, rdeg, diff in offenders:
        print(f"{ln:16s} : {ldeg:+7.2f}   ||   {rn:16s} : {rdeg:+7.2f}   --> |L|-|R| = {diff:+7.2f}")
    if offenders:
        top = offenders[0]
        print(f"\n>> TOP offender: {top[1]} vs {top[2]}  (|L|-|R|={top[0]:.2f} deg)")
    print("-" * 40)

def make_lr_pairs(hinge_table):
    """
    自动匹配左右关节对（名称以 L_ / R_ 开头）。
    返回 [(L_name, R_name), ...]
    """
    names = [rec['name'] for rec in hinge_table]
    name_set = set(names)
    pairs = []
    for nm in names:
        up = nm.upper()
        if up.startswith("L_"):
            rn = "R_" + nm[2:]
            if rn in name_set:
                pairs.append((nm, rn))
    uniq, seen = [], set()
    for p in pairs:
        key = tuple(sorted(p))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq

# ---------------------------
# 全局变量（切换pkl文件用）
# ---------------------------
pkl_files = []
pkl_index = 0
motion_data = {}
motion_data_keys = []
motion_id = 0
curr_start = 0
num_motions = 1
motion_acc = set()
time_step = 0
dt = 1/30
paused = False

def load_pkl_file(idx):
    """加载指定索引的pkl文件"""
    global motion_data, motion_data_keys, motion_id, time_step
    if 0 <= idx < len(pkl_files):
        print(f"\n[Loading] {pkl_files[idx]}")
        motion_data = joblib.load(pkl_files[idx])
        motion_data_keys = list(motion_data.keys())
        motion_id = 0  # 重置到第一个motion
        time_step = 0  # 重置时间
        print(f"Loaded {len(motion_data_keys)} motion(s): {motion_data_keys}")
        return True
    return False

# ---------------------------
# 键盘回调
# ---------------------------
def key_call_back(keycode):
    global pkl_index, motion_id, time_step, paused
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        paused = not paused
        print(f"Paused: {paused}")
    elif chr(keycode) == "T":
        pkl_index = (pkl_index + 1) % len(pkl_files)
        if load_pkl_file(pkl_index):
            print(f"Switched to file {pkl_index+1}/{len(pkl_files)}: {osp.basename(pkl_files[pkl_index])}")
    elif chr(keycode) == "Y":
        pkl_index = (pkl_index - 1) % len(pkl_files)
        if load_pkl_file(pkl_index):
            print(f"Switched to file {pkl_index+1}/{len(pkl_files)}: {osp.basename(pkl_files[pkl_index])}")
    elif chr(keycode) == "M":
        if len(motion_data_keys) > 1:
            motion_id = (motion_id + 1) % len(motion_data_keys)
            time_step = 0
            print(f"Switched to motion: {motion_data_keys[motion_id]}")
    else:
        print(f"Key not mapped: {chr(keycode)}")
        print("Controls: T=next file, Y=prev file, M=next motion in file, R=reset, Space=pause")

# ---------------------------
# 主程序
# ---------------------------
@hydra.main(version_base=None, config_path="../../phc/data/cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    # 关键修复：声明我们在 main 里会读写的全局变量，避免 time_step 变成本地变量
    global pkl_files, pkl_index, motion_data, motion_data_keys, motion_id, time_step, paused

    device = torch.device("cpu")
    humanoid_xml = cfg.robot.asset.assetFileName
    _ = SkeletonTree.from_mjcf(humanoid_xml)

    # 扫描文件夹获取所有pkl文件
    singles_dir = f"data/{cfg.robot.humanoid_type}/v1/singles"
    pkl_files = sorted(glob.glob(osp.join(singles_dir, "*.pkl")))

    if not pkl_files:
        print(f"[ERROR] No pkl files found in {singles_dir}")
        return

    print(f"[INFO] Found {len(pkl_files)} pkl files in {singles_dir}")

    # 如果指定了motion_name，尝试找到并设为初始文件
    if hasattr(cfg, 'motion_name'):
        target_file = osp.join(singles_dir, f"{cfg.motion_name}.pkl")
        if target_file in pkl_files:
            pkl_index = pkl_files.index(target_file)
            print(f"[INFO] Starting with specified file: {cfg.motion_name}.pkl")
        else:
            print(f"[WARN] Specified file {cfg.motion_name}.pkl not found, starting with first file")
            pkl_index = 0
    else:
        pkl_index = 0

    # 加载初始文件
    if not load_pkl_file(pkl_index):
        print("[ERROR] Failed to load initial pkl file")
        return

    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)

    # Box 可视化的持久状态
    BOX_GEOM_IDX = None
    BOX_RGBA = np.array([0.2, 0.7, 1.0, 0.35], dtype=np.float32)

    mj_model.opt.timestep = 1/30

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        # 预留一些几何给关节点显示
        for _ in range(50):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))

        # 构建 hinge 关节表 & 左右配对
        hinge_table, name2idx = build_hinge_joint_table(mj_model)
        lr_pairs = make_lr_pairs(hinge_table)

        # 简单缓冲：收集一小段时间窗做非对称统计
        angle_buffer = []
        buffer_window_frames = 60  # 约 2s (30Hz)

        last_dump_time = -1.0
        last_asym_time = -1.0

        print("\n=== Viewer Controls ===")
        print("T: Next pkl file")
        print("Y: Previous pkl file")
        print("M: Next motion in current file (if multiple)")
        print("R: Reset current motion")
        print("Space: Pause/Resume")
        print("=====================\n")

        while viewer.is_running():
            step_start = time.time()

            # 检查当前motion数据
            if not motion_data_keys:
                print("[WARN] No motion data loaded]")
                time.sleep(0.1)
                continue

            curr_motion_key = motion_data_keys[motion_id]
            curr_motion = motion_data[curr_motion_key]

            # 确保时间索引有效
            if 'dof_pos' not in curr_motion or curr_motion['dof_pos'].shape[0] == 0:
                print(f"[WARN] Invalid motion data for {curr_motion_key}")
                time.sleep(0.1)
                continue

            curr_time = int(time_step / mj_model.opt.timestep) % curr_motion['dof_pos'].shape[0]

            # ---- 更新机器人姿态 ----
            mj_data.qpos[:3] = curr_motion['root_trans_offset'][curr_time]
            # 数据里常是 xyzw，MuJoCo 需要 wxyz
            mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]]
            mj_data.qpos[7:] = curr_motion['dof_pos'][curr_time]

            mujoco.mj_forward(mj_model, mj_data)

            # 只在非暂停时更新时间
            if not paused:
                time_step += mj_model.opt.timestep

            # ---- 关节点云（SMPL joints）----
            if 'smpl_joints' in curr_motion:
                joint_gt = curr_motion['smpl_joints']
                max_pts = min(joint_gt.shape[1], viewer.user_scn.ngeom)
                for i in range(max_pts):
                    viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]

            # ---- 画 box ----
            try:
                bstate = curr_motion.get('box_state', None)
                bsize  = curr_motion.get('box_size', None)
                if bstate is not None:
                    center = np.asarray(bstate[:3], dtype=np.float32)
                    q_any  = np.asarray(bstate[3:], dtype=np.float32)
                    q_wxyz = _quat_xyzw_to_wxyz(q_any)
                    size_to_use = bsize if bsize is not None else np.array([0.5, 0.3, 0.3], dtype=np.float32)
                    if BOX_GEOM_IDX is None:
                        BOX_GEOM_IDX = add_visual_box(viewer.user_scn, center, size_to_use, q_wxyz, BOX_RGBA)
                    else:
                        update_visual_box(viewer.user_scn, BOX_GEOM_IDX, center, size_to_use, q_wxyz)
            except Exception:
                pass

            # ---- 收集当前帧各 hinge 角度 ----
            snap = {}
            for rec in hinge_table:
                nm, adr = rec['name'], rec['qpos_adr']
                snap[nm] = float(mj_data.qpos[adr])
            angle_buffer.append(snap)
            if len(angle_buffer) > buffer_window_frames:
                angle_buffer.pop(0)

            # ---- 定期打印角度信息 ----
            now = time.time()
            if last_dump_time < 0 or (now - last_dump_time) >= 1.0:
                dump_current_angles(mj_model, mj_data, hinge_table, only_legs=True)
                last_dump_time = now

            # ---- 定期打印非对称报告 ----
            if len(angle_buffer) == buffer_window_frames and (last_asym_time < 0 or (now - last_asym_time) >= 2.0):
                asymmetry_report(angle_buffer, lr_pairs)
                last_asym_time = now

            # ---- 同步渲染 ----
            viewer.sync()
            remain = mj_model.opt.timestep - (time.time() - step_start)
            if remain > 0:
                time.sleep(remain)

if __name__ == "__main__":
    main()
