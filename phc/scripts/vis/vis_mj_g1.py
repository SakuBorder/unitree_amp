"""
MuJoCo运动库可视化 - 修复坐标系版本
确保机器人正确站立
"""
import glob
import os
import sys
import pdb
import os.path as osp
import time
import math

sys.path.append(os.getcwd())

import joblib
import numpy as np

import mujoco
import mujoco.viewer
from phc.utils.motion_lib_g1 import MotionLibG1
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.flags import flags
import torch
from phc.utils.torch_g1_humanoid_batch import G1_ROTATION_AXIS


flags.test = True
flags.im_eval = True


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = torch.unbind(q1, -1)
    w2, x2, y2, z2 = torch.unbind(q2, -1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


R_IDENTITY = torch.eye(3)
QUAT_IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])

# --------------------------------------------------------- #

def add_visual_sphere(scene, pos, radius, rgba):

    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1],
                        mujoco.mjtGeom.mjGEOM_SPHERE, pos,
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    scene.geoms[scene.ngeom - 1].size[0] = radius


def add_visual_capsule(scene, point1, point2, radius, rgba):

    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom - 1],
                             mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                             point1[0], point1[1], point1[2],
                             point2[0], point2[1], point2[2])


class MuJoCoMotionVisualizer:
    def __init__(self):

        self.h1_xml = "resources/robots/g1/g1_29dof.xml"
        self.h1_urdf = "resources/robots/g1/g1_29dof.urdf"
        self.smpl_parents = [
            -1, 0, 1, 2,            # Pelvis->L_Hip->L_Knee->L_Ankle
            0, 4,                  #            R_Hip->R_Knee->R_Ankle
            0, 6, 7,               # Spine/Hips → Spine1 → Spine2
            8, 9, 10,              # Neck->Head
            8, 12, 13,             # L_Clavicle->L_Shoulder->L_Elbow
            14,                    # L_Wrist
            8, 16, 17,             # R_Clavicle->R_Shoulder->R_Elbow
            18                     # R_Wrist
        ]

        self.sk_tree = SkeletonTree.from_mjcf(self.h1_xml)


        self.motion_file = "data/g1/test.pkl"
        if not os.path.exists(self.motion_file):
            raise ValueError(f"Motion file {self.motion_file} not found")


        self.dt = 1.0 / 60.0
        self.time_step = 0.0
        self.speed_scale = 1.0
        self.paused = False


        self.motion_id = 0
        self.curr_start = 0
        self.num_motions = 1


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.show_axis = False
        self.show_joints = True
        self.show_smpl_joints = True  
        self.joint_radius = 0.05
        self.show_skeleton = True
        self.joint_geom_start = None        
        self.max_joint_geoms  = 32

        self.show_smpl_skeleton = True
        self.smpl_skeleton_color = np.array([1, 0, 0, 0.6], dtype=np.float32)
        self.smpl_skeleton_radius = 0.02


        self.setup_mujoco()
        self.setup_motion_lib()

    def setup_mujoco(self):
        self.mj_model = mujoco.MjModel.from_xml_path(self.h1_xml)
        self.mj_model.opt.timestep = self.dt
        self.mj_data = mujoco.MjData(self.mj_model)
        self.num_dofs = self.mj_model.nq - 7

    def setup_motion_lib(self):
        self.motion_lib = MotionLibG1(
            motion_file=self.motion_file,
            device=self.device,
            masterfoot_conifg=None,
            fix_height=True,
            multi_thread=False,
            mjcf_file=self.h1_xml,
        )
        self.motion_lib.load_motions(
            skeleton_trees=[self.sk_tree] * self.num_motions,
            gender_betas=[torch.zeros(17)] * self.num_motions,
            limb_weights=[np.zeros(10)] * self.num_motions,
            random_sample=False,
        )
        self.motion_keys = self.motion_lib.curr_motion_keys

    def key_callback(self, keycode):
        """键盘回调函数"""
        key_char = chr(keycode)
        if keycode == 263:  # 左箭头
            self.motion_id = (self.motion_id - 1) % self.num_motions
            self.time_step = 0.0
            print(f"Motion ID: {self.motion_id}")
        elif keycode == 262:  # 右箭头
            self.motion_id = (self.motion_id + 1) % self.num_motions
            self.time_step = 0.0
            print(f"Motion ID: {self.motion_id}")
        elif key_char == " ":
            self.paused = not self.paused
            print(f"Paused: {self.paused}")
        elif key_char == "+":
            self.speed_scale = min(self.speed_scale * 1.2, 5.0)
            print(f"Speed scale: {self.speed_scale:.2f}")
        elif key_char == "-":
            self.speed_scale = max(self.speed_scale / 1.2, 0.1)
            print(f"Speed scale: {self.speed_scale:.2f}")
        elif key_char == "S":
            self.show_skeleton = not self.show_skeleton
            print(f"Show skeleton: {self.show_skeleton}")
        elif key_char == "J":
            self.show_joints = not self.show_joints
            print(f"Show joints: {self.show_joints}")
        elif key_char == "M":
            self.show_smpl_joints = not self.show_smpl_joints
            print(f"Show SMPL joints: {self.show_smpl_joints}")
        elif key_char == "R":
            self.time_step = 0
            print("Reset time")
        elif key_char == "K":
            self.show_smpl_skeleton = not self.show_smpl_skeleton
            print(f"Show SMPL skeleton: {self.show_smpl_skeleton}")

    def update_robot_state(self):


        motion_len  = self.motion_lib.get_motion_length(self.motion_id).item()
        motion_time = self.time_step % motion_len

        res = self.motion_lib.get_motion_state(
            torch.tensor([self.motion_id], device=self.device),
            torch.tensor([motion_time],    device=self.device),
        )

        root_pos = res["root_pos"][0]      # (x,y,z)
        root_rot = res["root_rot"][0]      # (x,y,z,w)  or  (w,x,y,z)
        dof_pos  = res["dof_pos"][0]


        root_rot = root_rot[[3, 0, 1, 2]]
        self.mj_data.qpos[:3]   = root_pos.cpu().numpy()
        self.mj_data.qpos[3:7]  = root_rot.cpu().numpy()

        n = min(dof_pos.shape[0], self.num_dofs)
        self.mj_data.qpos[7:7+n] = dof_pos[:n].cpu().numpy()

        mujoco.mj_forward(self.mj_model, self.mj_data)
        return res


    def update_visualization(self, viewer, res):
        # ------- 更新 SMPL（红球）-------
        if self.show_smpl_joints and "smpl_joints" in res:
            # 移除调试打印
            # print(f"SMPL joints shape: {res['smpl_joints'].shape}")
            # print(f"First joint position: {res['smpl_joints'][0][0]}")
            
            # 处理不同可能的数据格式
            sj = res["smpl_joints"]
            if len(sj.shape) == 3 and sj.shape[0] == 1:
                sj = sj[0]  # 从 [1, 24, 3] 变为 [24, 3]
            
            n_s = min(len(sj), self.max_smpl_joints)
            
            # 更新红色球的位置
            for i in range(n_s):
                geom_id = self.smpl_geom_start + i
                if torch.is_tensor(sj[i]):
                    viewer.user_scn.geoms[geom_id].pos[:] = sj[i].cpu().numpy()
                else:
                    viewer.user_scn.geoms[geom_id].pos[:] = sj[i]

        # ------- 更新机器人关节（蓝球）-------
        if self.show_joints and "rg_pos" in res:
            rg_pos = res["rg_pos"][0]
            n_r = min(len(rg_pos), self.max_robot_joints)
            for i in range(n_r):
                geom_id = self.robot_geom_start + i
                viewer.user_scn.geoms[geom_id].pos[:] = rg_pos[i].cpu().numpy()

        # 清除所有动态绘制的线条
        viewer.user_scn.ngeom = self.robot_geom_start + self.max_robot_joints

        # ------- 只绘制机器人骨架（绿色）-------
        if self.show_skeleton and "rg_pos" in res and hasattr(self.sk_tree, 'parent_indices'):
            rg_pos = res["rg_pos"][0]
            n_r = min(len(rg_pos), self.max_robot_joints)
            parents = self.sk_tree.parent_indices
            for i, p in enumerate(parents):
                if p >= 0 and i < n_r and p < n_r:
                    add_visual_capsule(
                        viewer.user_scn,
                        rg_pos[p].cpu().numpy(),
                        rg_pos[i].cpu().numpy(),
                        0.02,
                        np.array([0, 1, 0, 0.5], dtype=np.float32)
                    )

    def run(self):
        """运行可视化"""
        print("MuJoCo Motion Visualization")
        print("控制键：")
        print("  左/右箭头 - 切换动作")
        print("  空格 - 暂停/继续")
        print("  +/- - 速度控制")
        print("  S - 显示/隐藏骨架")
        print("  J - 显示/隐藏机器人关节")
        print("  M - 显示/隐藏SMPL关节（红球）")
        print("  K - 显示/隐藏SMPL骨架（未使用）")
        print("  R - 重置时间")
        
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data,
                                        key_callback=self.key_callback) as viewer:

            # ------- ① 为 SMPL 关节预留红球 -------
            self.smpl_geom_start = viewer.user_scn.ngeom
            self.max_smpl_joints = 24  # SMPL有24个关节
            for _ in range(self.max_smpl_joints):
                add_visual_sphere(viewer.user_scn, np.zeros(3),
                                self.joint_radius,
                                np.array([1, 0, 0, 1], dtype=np.float32))  # 红

            # ------- ② 为机器人关节预留蓝球 -------
            self.robot_geom_start = viewer.user_scn.ngeom
            self.max_robot_joints = 32
            for _ in range(self.max_robot_joints):
                add_visual_sphere(viewer.user_scn, np.zeros(3),
                                self.joint_radius,
                                np.array([0, 0, 1, 1], dtype=np.float32))  # 蓝
                                
            while viewer.is_running():
                t0 = time.time()
                res = self.update_robot_state()
                self.update_visualization(viewer, res)
                if not self.paused:
                    self.time_step += self.dt * self.speed_scale
                viewer.sync()
                lag = self.dt - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)


def main():
    try:
        vis = MuJoCoMotionVisualizer()
        vis.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()