"""
MuJoCo Taihu Robot Motion Visualization - 修正版本
修复坐标系对齐和视角问题
"""
import os
import sys
import time
import numpy as np

import mujoco
import mujoco.viewer
from phc.utils.motion_lib_taihu import MotionLibTaihu  # 假设 Taihu 与 G1 接口兼容
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.flags import flags
import torch

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

def add_visual_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1], mujoco.mjtGeom.mjGEOM_SPHERE,
                        pos, np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    scene.geoms[scene.ngeom - 1].size[0] = radius

def add_visual_capsule(scene, point1, point2, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1], mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.zeros(3), np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom - 1], mujoco.mjtGeom.mjGEOM_CAPSULE,
                             radius, point1[0], point1[1], point1[2],
                             point2[0], point2[1], point2[2])

class TaihuMotionVisualizer:
    def __init__(self):
        self.model_path = "resources/robots/taihu/taihu.xml"
        self.motion_file = "data/taihu/0-Male2Walking_c3d_B15 -  Walk turn around_poses.pkl"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dt = 1.0 / 60.0
        self.time_step = 0.0
        self.speed_scale = 1.0
        self.paused = False
        
        # 显示控制参数
        self.show_joints = True
        self.show_skeleton = True
        self.show_smpl_joints = True
        self.joint_radius = 0.05
        self.max_joints = 32
        self.max_smpl_joints = 24
        
        self.setup_model()
        self.setup_motion()

    def setup_model(self):
        self.mj_model = mujoco.MjModel.from_xml_path(self.model_path)
        self.mj_model.opt.timestep = self.dt
        self.mj_data = mujoco.MjData(self.mj_model)
        self.skeleton = SkeletonTree.from_mjcf(self.model_path)
        self.num_dofs = self.mj_model.nq - 7

    def setup_motion(self):
        self.motion_lib = MotionLibTaihu(
            motion_file=self.motion_file,
            device=self.device,
            mjcf_file=self.model_path,
            fix_height=True
        )
        self.motion_lib.load_motions(
            skeleton_trees=[self.skeleton],
            gender_betas=[torch.zeros(17)],
            limb_weights=[np.zeros(10)],
            random_sample=False
        )

    def update_state(self):
        motion_time = self.time_step % self.motion_lib.get_motion_length(0).item()
        res = self.motion_lib.get_motion_state(
            torch.tensor([0], device=self.device),
            torch.tensor([motion_time], device=self.device)
        )
        
        # 获取运动数据
        root_pos = res["root_pos"][0]  # (x,y,z)
        root_rot = res["root_rot"][0]  # 可能是 (x,y,z,w) 或 (w,x,y,z)
        dof_pos = res["dof_pos"][0]
        
        # 关键修正：确保四元数格式正确 - 转换为 MuJoCo 的 (w,x,y,z) 格式
        if root_rot.shape[0] == 4:
            # 假设输入是 (x,y,z,w) 格式，转换为 (w,x,y,z)
            root_rot = root_rot[[3, 0, 1, 2]]
        
        # 设置机器人状态
        self.mj_data.qpos[:3] = root_pos.cpu().numpy()
        self.mj_data.qpos[3:7] = root_rot.cpu().numpy()
        
        # 设置关节角度，确保不超出自由度范围
        n_dofs = min(dof_pos.shape[0], self.num_dofs)
        self.mj_data.qpos[7:7+n_dofs] = dof_pos[:n_dofs].cpu().numpy()
        
        # 前向计算
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return res

    def update_viewer(self, viewer, res):
        # 重置几何体计数到机器人关节开始位置
        viewer.user_scn.ngeom = self.robot_geom_start + self.max_joints
        
        # 更新 SMPL 关节可视化（红色球）
        if self.show_smpl_joints and "smpl_joints" in res:
            sj = res["smpl_joints"]
            if len(sj.shape) == 3 and sj.shape[0] == 1:
                sj = sj[0]  # 从 [1, 24, 3] 变为 [24, 3]
            
            n_s = min(len(sj), self.max_smpl_joints)
            for i in range(n_s):
                geom_id = self.smpl_geom_start + i
                if torch.is_tensor(sj[i]):
                    viewer.user_scn.geoms[geom_id].pos[:] = sj[i].cpu().numpy()
                else:
                    viewer.user_scn.geoms[geom_id].pos[:] = sj[i]
        
        # 更新机器人关节可视化（蓝色球）
        if self.show_joints and "rg_pos" in res:
            rg_pos = res["rg_pos"][0]
            n_r = min(len(rg_pos), self.max_joints)
            for i in range(n_r):
                geom_id = self.robot_geom_start + i
                viewer.user_scn.geoms[geom_id].pos[:] = rg_pos[i].cpu().numpy()
        
        # 绘制机器人骨架（绿色线条）
        if self.show_skeleton and "rg_pos" in res and hasattr(self.skeleton, 'parent_indices'):
            rg_pos = res["rg_pos"][0]
            n_r = min(len(rg_pos), self.max_joints)
            parents = self.skeleton.parent_indices
            
            for i, p in enumerate(parents):
                if p >= 0 and i < n_r and p < n_r:
                    add_visual_capsule(
                        viewer.user_scn,
                        rg_pos[p].cpu().numpy(),
                        rg_pos[i].cpu().numpy(),
                        0.02,
                        np.array([0, 1, 0, 0.5], dtype=np.float32)  # 绿色半透明
                    )

    def key_callback(self, keycode):
        """键盘控制回调"""
        key_char = chr(keycode) if keycode < 256 else ""
        
        if key_char == " ":
            self.paused = not self.paused
            print(f"暂停: {self.paused}")
        elif key_char == "+" or key_char == "=":
            self.speed_scale = min(self.speed_scale * 1.2, 5.0)
            print(f"速度倍率: {self.speed_scale:.2f}")
        elif key_char == "-":
            self.speed_scale = max(self.speed_scale / 1.2, 0.1)
            print(f"速度倍率: {self.speed_scale:.2f}")
        elif key_char == "R" or key_char == "r":
            self.time_step = 0.0
            print("重置时间")
        elif key_char == "S" or key_char == "s":
            self.show_skeleton = not self.show_skeleton
            print(f"显示骨架: {self.show_skeleton}")
        elif key_char == "J" or key_char == "j":
            self.show_joints = not self.show_joints
            print(f"显示机器人关节: {self.show_joints}")
        elif key_char == "M" or key_char == "m":
            self.show_smpl_joints = not self.show_smpl_joints
            print(f"显示SMPL关节: {self.show_smpl_joints}")

    def setup_camera(self, viewer):
        """设置更好的相机视角"""
        # 设置相机距离和角度，参考成功的实现
        cam = viewer.cam
        cam.distance = 4.0  # 相机距离
        cam.elevation = -20  # 仰角
        cam.azimuth = 45    # 方位角
        cam.lookat[0] = 0.0  # 看向原点
        cam.lookat[1] = 0.0
        cam.lookat[2] = 1.0  # 稍微抬高视点

    def run(self):
        print("== Taihu MuJoCo Motion Visualizer - 修正版 ==")
        print("控制键:")
        print("  空格键 - 暂停/继续")
        print("  +/- - 调整播放速度")
        print("  R - 重置时间")
        print("  S - 显示/隐藏机器人骨架")
        print("  J - 显示/隐藏机器人关节")
        print("  M - 显示/隐藏SMPL关节")
        
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data,
                                          key_callback=self.key_callback) as viewer:
            
            # 设置更好的相机视角
            self.setup_camera(viewer)
            
            # 为 SMPL 关节预留红色球体
            self.smpl_geom_start = viewer.user_scn.ngeom
            for _ in range(self.max_smpl_joints):
                add_visual_sphere(viewer.user_scn, np.zeros(3), self.joint_radius,
                                  np.array([1, 0, 0, 1], dtype=np.float32))  # 红色
            
            # 为机器人关节预留蓝色球体
            self.robot_geom_start = viewer.user_scn.ngeom
            for _ in range(self.max_joints):
                add_visual_sphere(viewer.user_scn, np.zeros(3), self.joint_radius,
                                  np.array([0, 0, 1, 1], dtype=np.float32))  # 蓝色
            
            # 主循环
            while viewer.is_running():
                start_time = time.time()
                
                # 更新机器人状态
                res = self.update_state()
                
                # 更新可视化
                self.update_viewer(viewer, res)
                
                # 时间步进
                if not self.paused:
                    self.time_step += self.dt * self.speed_scale
                
                # 同步显示
                viewer.sync()
                
                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = max(0, self.dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

def main():
    try:
        vis = TaihuMotionVisualizer()
        vis.run()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()