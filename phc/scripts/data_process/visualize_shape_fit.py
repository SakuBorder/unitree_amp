import torch
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from phc.utils.torch_humanoid_batch import Humanoid_Batch
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from easydict import EasyDict
import numpy as np
from scipy.spatial.transform import Rotation as sRot

def visualize_fitted_shape(humanoid_type="unitree_g1_fitting", robot_cfg_path="phc/data/cfg/robot/unitree_g1_fitting.yaml"):
    # 加载配置
    import yaml
    with open(robot_cfg_path, "r") as f:
        robot_cfg = EasyDict(yaml.safe_load(f))

    # 加载优化结果
    shape_new, scale = joblib.load(f"data/{humanoid_type}/shape_optimized_v1.pkl")

    # SMPL Parser
    smpl_parser = SMPL_Parser("data/smpl", gender="neutral")

    # 构建SMPL站立姿态
    pose_aa_stand = np.zeros((1, 72)).reshape(-1, 24, 3)
    for modifiers in robot_cfg.smpl_pose_modifier:
        key, value = list(modifiers.items())[0]
        pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index(key)] = sRot.from_euler("xyz", eval(value), degrees=False).as_rotvec()
    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72)).float()

    trans = torch.zeros([1, 3])
    verts, joints = smpl_parser.get_joints_verts(pose_aa_stand, shape_new, trans)
    joints = (joints - joints[:, 0]) * scale + joints[:, 0]

    # Humanoid FK
    humanoid_fk = Humanoid_Batch(robot_cfg)
    pose_aa_robot = torch.zeros([1, 1, humanoid_fk.num_bodies, 3])
    offset = joints[:, 0] - trans
    root_trans_offset = trans + offset
    fk_return = humanoid_fk.fk_batch(pose_aa_robot, root_trans_offset)

    # 读取匹配关节索引
    robot_joint_names_augment = humanoid_fk.body_names_augment
    robot_joint_pick = [i[0] for i in robot_cfg.joint_matches]
    smpl_joint_pick = [i[1] for i in robot_cfg.joint_matches]
    robot_joint_pick_idx = [robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    j3d = fk_return.global_translation_extend[0, :, robot_joint_pick_idx, :].detach().numpy()
    j3d -= j3d[:, 0:1]
    j3d_joints = joints[0, smpl_joint_pick_idx].detach().numpy()
    j3d_joints -= j3d_joints[0:1]

    # 可视化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(90, 0)
    ax.scatter(j3d[:, 0], j3d[:, 1], j3d[:, 2], label='Humanoid Shape', c='blue')
    ax.scatter(j3d_joints[:, 0], j3d_joints[:, 1], j3d_joints[:, 2], label='Fitted SMPL', c='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.legend()
    plt.title("Shape Fit Visualization")
    plt.show()

if __name__ == "__main__":
    visualize_fitted_shape()
