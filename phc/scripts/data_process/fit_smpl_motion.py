# sitecustomize.py
import numpy as np

# 兼容旧 NumPy 别名（仅当前进程）
_aliases = [
    ("bool", bool), ("int", int), ("float", float), ("complex", complex),
    ("object", object), ("unicode", str), ("str", str),
]
for name, typ in _aliases:
    if not hasattr(np, name):
        try:
            setattr(np, name, typ)
        except Exception:
            pass

import glob
import os
import sys
import os.path as osp
import joblib
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from easydict import EasyDict
import hydra
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as sRot

# 项目内依赖
sys.path.append(os.getcwd())
from smpl_sim.utils import torch_utils
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from smpl_sim.utils.smoothing_utils import gaussian_filter_1d_batch
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from phc.phc.utils.torch_humanoid_batch import Humanoid_Batch


# -----------------------------
# helpers（简化）
# -----------------------------
def _fk_body_names(humanoid_fk):
    if hasattr(humanoid_fk, "body_names_augment") and humanoid_fk.body_names_augment:
        return list(humanoid_fk.body_names_augment)
    return list(humanoid_fk.body_names)

def _fk_global_translation(fk_ret):
    if hasattr(fk_ret, "global_translation_extend"):
        return fk_ret.global_translation_extend
    return fk_ret.global_translation

def _load_amass_npz(path):
    entry = dict(np.load(open(path, "rb"), allow_pickle=True))
    if "mocap_framerate" not in entry and "mocap_frame_rate" not in entry:
        return None
    fps = entry.get("mocap_framerate", entry.get("mocap_frame_rate"))
    trans = entry["trans"]
    pose_aa = np.concatenate([entry["poses"][:, :66], np.zeros((trans.shape[0], 6))], axis=-1)
    return {"pose_aa": pose_aa, "trans": trans, "fps": fps}


# -----------------------------
# core（加入 L_HIP_Y_S / L_HIP_R_S / L_ANKLE_P_S 固定补偿）
# -----------------------------
def process_motion(keys, key2file, cfg):
    device = torch.device("cpu")
    humanoid_fk = Humanoid_Batch(cfg.robot)

    expected_feet = list(getattr(cfg.robot, "expected_foot_body_names", ["L_ANKLE_R_S", "R_ANKLE_R_S"]))

    body_names_aug = _fk_body_names(humanoid_fk)
    robot_pick_names = [i[0] for i in cfg.robot.joint_matches]
    smpl_pick_names  = [i[1] for i in cfg.robot.joint_matches]
    robot_pick_idx   = [body_names_aug.index(n) for n in robot_pick_names]
    smpl_pick_idx    = [SMPL_BONE_ORDER_NAMES.index(n) for n in smpl_pick_names]

    # —— 通过 body_names 顺序映射到 dof 索引（去掉 base_link，所以 -1）
    lhy_body_idx = body_names_aug.index("L_HIP_Y_S")  if "L_HIP_Y_S"  in body_names_aug else None
    lhr_body_idx = body_names_aug.index("L_HIP_R_S")  if "L_HIP_R_S"  in body_names_aug else None
    lap_body_idx = body_names_aug.index("L_ANKLE_P_S")if "L_ANKLE_P_S"in body_names_aug else None
    lhy_dof_idx = (lhy_body_idx - 1) if lhy_body_idx is not None else None
    lhr_dof_idx = (lhr_body_idx - 1) if lhr_body_idx is not None else None
    lap_dof_idx = (lap_body_idx - 1) if lap_body_idx is not None else None

    # —— 读取补偿角（度）；按你的轴向需要调整符号
    lhy_comp_deg = float(getattr(cfg, "l_hip_yaw_comp_deg",   -18.0))  # 外旋示例
    lhr_comp_deg = float(getattr(cfg, "l_hip_roll_comp_deg",   +3.0))  # 轻微外展/内收补偿
    lap_comp_deg = float(getattr(cfg, "l_ankle_pitch_comp_deg",-15.0)) # 跖/背屈补偿
    lhy_comp_rad = lhy_comp_deg * np.pi / 180.0
    lhr_comp_rad = lhr_comp_deg * np.pi / 180.0
    lap_comp_rad = lap_comp_deg * np.pi / 180.0

    # —— 打印补偿信息和轴向，便于判断方向（debug）
    if lhy_dof_idx is not None:
        print(f"[INFO] L_HIP_Y_S comp = {lhy_comp_deg:+.2f} deg (dof_idx={lhy_dof_idx})")
        try:
            ax = humanoid_fk.dof_axis[lhy_dof_idx]
            if torch.is_tensor(ax): ax = ax.squeeze().detach().cpu().numpy()
            ax = np.asarray(ax); print(f"[INFO] L_HIP_Y axis ~ ({ax[0]:+.3f}, {ax[1]:+.3f}, {ax[2]:+.3f})")
        except Exception: pass
    else:
        print("[WARN] L_HIP_Y_S not found; skip L hip yaw compensation.")

    if lhr_dof_idx is not None:
        print(f"[INFO] L_HIP_R_S comp = {lhr_comp_deg:+.2f} deg (dof_idx={lhr_dof_idx})")
        try:
            ax = humanoid_fk.dof_axis[lhr_dof_idx]
            if torch.is_tensor(ax): ax = ax.squeeze().detach().cpu().numpy()
            ax = np.asarray(ax); print(f"[INFO] L_HIP_R axis ~ ({ax[0]:+.3f}, {ax[1]:+.3f}, {ax[2]:+.3f})")
        except Exception: pass
    else:
        print("[WARN] L_HIP_R_S not found; skip L hip roll compensation.")

    if lap_dof_idx is not None:
        print(f"[INFO] L_ANKLE_P_S comp = {lap_comp_deg:+.2f} deg (dof_idx={lap_dof_idx})")
        try:
            ax = humanoid_fk.dof_axis[lap_dof_idx]
            if torch.is_tensor(ax): ax = ax.squeeze().detach().cpu().numpy()
            ax = np.asarray(ax); print(f"[INFO] L_ANKLE_P axis ~ ({ax[0]:+.3f}, {ax[1]:+.3f}, {ax[2]:+.3f})")
        except Exception: pass
    else:
        print("[WARN] L_ANKLE_P_S not found; skip L ankle pitch compensation.")

    # —— SMPL & 形状
    smpl = SMPL_Parser(model_path="data/smpl", gender="neutral")
    shape_new, scale = joblib.load(f"data/{cfg.robot.humanoid_type}/shape_optimized_v1.pkl")

    all_data = {}
    pbar = tqdm(keys, position=0, leave=True)
    for k in pbar:
        amass = _load_amass_npz(key2file[k])
        if amass is None: continue

        skip = max(1, int(amass["fps"] // 30))
        trans = torch.from_numpy(amass["trans"][::skip]).float()
        pose_aa_smpl = torch.from_numpy(amass["pose_aa"][::skip]).float()
        N = trans.shape[0]
        if N < 10:
            print(f"[WARN] {k}: too short, skip"); continue

        with torch.no_grad():
            verts, joints = smpl.get_joints_verts(pose_aa_smpl, shape_new, trans)
            root_pos = joints[:, 0:1]
            joints = (joints - root_pos) * scale.detach() + root_pos
        ground_z = verts[0, :, 2].min().item()
        joints[..., 2] -= ground_z

        offset = joints[:, 0] - trans
        root_trans_offset = (trans + offset).clone()

        gt_root_quat = torch.from_numpy(
            (sRot.from_rotvec(pose_aa_smpl[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()
        ).float()
        gt_root_rot = torch.from_numpy(
            sRot.from_quat(torch_utils.calc_heading_quat(gt_root_quat)).as_rotvec()
        ).float()

        dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))
        dof_pos_new   = Variable(dof_pos.clone(), requires_grad=True)
        root_rot_new  = Variable(gt_root_rot.clone(), requires_grad=True)
        root_pos_offset = Variable(torch.zeros(1, 3), requires_grad=True)
        optim = torch.optim.Adam([dof_pos_new, root_rot_new, root_pos_offset], lr=0.02)

        kernel_size, sigma = 5, 0.75
        fit_iters = int(cfg.get("fitting_iterations", 500))

        for it in range(fit_iters):
            pose_aa_h1 = torch.cat(
                [root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new],
                dim=2
            )
            fk_ret = humanoid_fk.fk_batch(
                pose_aa_h1, root_trans_offset[None, :] + root_pos_offset
            )
            gtr = _fk_global_translation(fk_ret)[0]

            diff = gtr[:, robot_pick_idx] - joints[:, smpl_pick_idx]
            loss = diff.norm(dim=-1).mean() + 0.01 * torch.mean(torch.square(dof_pos_new))

            optim.zero_grad(); loss.backward(); optim.step()

            lo, hi = humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None]
            dof_pos_new.data.clamp_(lo, hi)
            dof_pos_new.data = gaussian_filter_1d_batch(
                dof_pos_new.squeeze().transpose(1, 0)[None,], kernel_size, sigma
            ).transpose(2, 1)[..., None]

            if it % 100 == 0:
                pbar.set_description_str(f"{k} | it {it}/{fit_iters} | loss {loss.item():.4f}")

        # —— 保存前补偿（统一作用于所有帧），并再次 clamp
        with torch.no_grad():
            if lhy_dof_idx is not None:
                dof_pos_new.data[0, :, lhy_dof_idx, 0] += lhy_comp_rad
            if lhr_dof_idx is not None:
                dof_pos_new.data[0, :, lhr_dof_idx, 0] += lhr_comp_rad
            if lap_dof_idx is not None:
                dof_pos_new.data[0, :, lap_dof_idx, 0] += lap_comp_rad
            # 夹紧
            lo, hi = humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None]
            dof_pos_new.data.clamp_(lo, hi)

        if lhy_dof_idx is not None:
            print(f"[APPLY] L_HIP_Y_S += {lhy_comp_deg:+.2f} deg")
        if lhr_dof_idx is not None:
            print(f"[APPLY] L_HIP_R_S += {lhr_comp_deg:+.2f} deg")
        if lap_dof_idx is not None:
            print(f"[APPLY] L_ANKLE_P_S += {lap_comp_deg:+.2f} deg")

        pose_aa_h1_final = torch.cat(
            [root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new],
            dim=2
        )

        root_trans_offset_dump = (root_trans_offset + root_pos_offset).clone()
        height_diff = -0.50
        root_trans_offset_dump[..., 2] -= height_diff

        joints_dump = joints.detach().cpu().numpy().copy()
        joints_dump[..., 2] -= height_diff

        fk_seq = humanoid_fk.fk_batch(
            pose_aa_h1_final.detach(), root_trans_offset_dump[None, :].detach()
        )
        gtr_seq = _fk_global_translation(fk_seq)[0]
        body_names = _fk_body_names(humanoid_fk)

        foot_idx = []
        for n in expected_feet:
            if n not in body_names:
                print(f"[WARN] Foot '{n}' not in body names, using fallback…")
                foot_idx = [0, 1]; break
            foot_idx.append(body_names.index(n))

        if foot_idx:
            feet_world = gtr_seq[:, foot_idx, :]
            root_world = root_trans_offset_dump
            feet_rel = feet_world - root_world[:, None, :]
            foot_positions_flat = feet_rel.reshape(feet_rel.shape[0], -1).detach().cpu().numpy()
        else:
            foot_positions_flat = np.zeros((N, 6))

        data_dump = {
            "root_trans_offset": root_trans_offset_dump.squeeze().detach().cpu().numpy(),
            "pose_aa":           pose_aa_h1_final.squeeze().detach().cpu().numpy(),   # ← 修正：.cpu().numpy()
            "dof_pos":           dof_pos_new.squeeze().detach().cpu().numpy(),        # 已含三处补偿
            "dof":               dof_pos_new.squeeze().detach().cpu().numpy(),
            "root_rot":          sRot.from_rotvec(root_rot_new.detach().cpu().numpy()).as_quat(),
            "smpl_joints":       joints_dump,
            "fps":               30,
            "foot_positions":    foot_positions_flat,
            "foot_body_names":   expected_feet if foot_idx else [],
        }

        all_data[k] = data_dump

    return all_data


# -----------------------------
# entry
# -----------------------------
@hydra.main(version_base=None, config_path="../../phc/data/cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    if "amass_root" not in cfg:
        raise ValueError("amass_root is not specified in the config")

    all_npz = glob.glob(osp.join(cfg.amass_root, "**/*.npz"), recursive=True)
    base_len = len(cfg.amass_root.split("/"))
    key2file = {"0-" + "_".join(p.split("/")[base_len:]).replace(".npz", ""): p for p in all_npz}
    keys = list(key2file.keys())
    if not cfg.get("fit_all", False):
        keys = ["0-Female1Walking_c3d_B3 - walk1_poses"]

    torch.set_num_threads(1)
    mp.set_sharing_strategy("file_descriptor")

    all_data = process_motion(keys, key2file, cfg)

    save_dir = f"data/{cfg.robot.humanoid_type}/v1/singles"
    os.makedirs(save_dir, exist_ok=True)
    for k, v in all_data.items():
        out = osp.join(save_dir, f"{k}.pkl")
        joblib.dump({k: v}, out)
        print(f"[SAVE] {out}")

    print(f"[DONE] {len(all_data)} motions saved to {save_dir}")


if __name__ == "__main__":
    main()
