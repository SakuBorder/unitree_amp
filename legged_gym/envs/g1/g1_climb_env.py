import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

class G1ClimbRobot(G1Robot):
    """G1攀爬任务环境"""
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 攀爬任务需要2个actors: robot + climb_object
        self.actors_per_env = 2
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
    def _create_envs(self):
        """创建环境，包含机器人和攀爬物体"""
        # 先创建攀爬物体资产
        self._create_climb_assets()
        super()._create_envs()
        
    def _create_climb_assets(self):
        """创建攀爬物体资产"""
        self.climb_assets = []
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.density = 1000.0
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        
        # 创建不同类型的攀爬物体
        for cat in self.cfg.env.objCategories:
            if cat == "stairs":
                # 创建楼梯
                asset = self._create_stairs_asset(asset_options)
            elif cat == "ramp":
                # 创建斜坡
                asset = self._create_ramp_asset(asset_options)
            elif cat == "obstacle":
                # 创建障碍物
                asset = self._create_obstacle_asset(asset_options)
            else:
                # 默认创建一个箱子
                asset = self.gym.create_box(self.sim, 1.0, 0.5, 0.3, asset_options)
            self.climb_assets.append(asset)
            
    def _create_stairs_asset(self, options):
        """创建楼梯资产"""
        # 简化版：用多个箱子组成楼梯
        stair_width = 1.5
        stair_depth = 0.3
        stair_height = 0.15
        num_steps = 4
        
        # 这里简化处理，实际应该组合多个箱子
        return self.gym.create_box(self.sim, stair_width, stair_depth * num_steps, stair_height * num_steps, options)
        
    def _create_ramp_asset(self, options):
        """创建斜坡资产"""
        ramp_length = 2.0
        ramp_width = 1.5
        ramp_height = 0.5
        return self.gym.create_box(self.sim, ramp_width, ramp_length, ramp_height, options)
        
    def _create_obstacle_asset(self, options):
        """创建障碍物资产"""
        return self.gym.create_box(self.sim, 1.0, 1.0, 0.4, options)
        
    def _build_env(self, env_id, env_ptr, robot_asset):
        """构建单个环境"""
        super()._build_env(env_id, env_ptr, robot_asset)
        
        # 添加攀爬物体
        climb_asset_id = env_id % len(self.climb_assets)
        climb_asset = self.climb_assets[climb_asset_id]
        
        climb_pose = gymapi.Transform()
        climb_pose.p = gymapi.Vec3(2.0, 0.0, 0.0)  # 放在机器人前方
        climb_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        climb_handle = self.gym.create_actor(
            env_ptr, climb_asset, climb_pose,
            f"climb_object_{env_id}", env_id, 0, 0
        )
        
        # 设置颜色
        self.gym.set_rigid_body_color(
            env_ptr, climb_handle, 0, 
            gymapi.MESH_VISUAL,
            gymapi.Vec3(0.5, 0.5, 0.5)
        )
        
    def _init_buffers(self):
        """初始化缓冲区"""
        super()._init_buffers()
        
        # 攀爬物体状态
        self.climb_object_states = self.all_root_states[:, 1]  # [num_envs, 13]
        self.climb_object_pos = self.climb_object_states[:, 0:3]
        self.climb_object_rot = self.climb_object_states[:, 3:7]
        
        # 目标攀爬位置（物体顶部）
        self.climb_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.climb_target_pos[:, 2] = 0.5  # 默认高度
        
        # 攀爬进度跟踪
        self.climb_progress = torch.zeros(self.num_envs, device=self.device)
        self.prev_climb_height = torch.zeros(self.num_envs, device=self.device)
        
        # IET（交互早期终止）相关
        if self.cfg.env.enableIET:
            self.IET_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.IET_triggered = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            
        # 物体边界框（简化版）
        self.object_bbox_size = torch.tensor([1.5, 2.0, 0.5], device=self.device)  # 宽x深x高
        
    def compute_observations(self):
        """计算观测"""
        super().compute_observations()
        
        if self.cfg.env.enableTaskObs:
            task_obs = self._compute_task_obs()
            self.obs_buf = torch.cat([self.obs_buf, task_obs], dim=-1)
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, task_obs], dim=-1)
                
    def _compute_task_obs(self):
        """计算任务相关观测"""
        # 计算相对位置
        rel_object_pos = self.climb_object_pos - self.base_pos
        rel_target_pos = self.climb_target_pos - self.base_pos
        
        # 将相对位置转换到机器人局部坐标系
        heading_rot = torch_utils.calc_heading_quat_inv(self.base_quat)
        local_object_pos = quat_rotate(heading_rot, rel_object_pos)
        local_target_pos = quat_rotate(heading_rot, rel_target_pos)
        
        # 物体朝向（简化为四元数的一部分）
        object_rot_obs = self.climb_object_rot
        
        # 计算物体8个角点的位置（边界框）
        bbox_points = self._compute_bbox_points()
        local_bbox_points = self._transform_bbox_to_local(bbox_points, heading_rot)
        
        # 组合观测
        task_obs = torch.cat([
            local_target_pos,                           # 3
            local_object_pos,                           # 3
            object_rot_obs,                             # 4
            local_bbox_points.reshape(self.num_envs, -1)[:, :24]  # 24 (8个点的前8个坐标)
        ], dim=-1)
        
        return task_obs
        
    def _compute_bbox_points(self):
        """计算物体边界框的8个角点"""
        half_size = self.object_bbox_size / 2
        
        # 8个角点的局部坐标
        corners = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # 底部4个点
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]        # 顶部4个点
        ], device=self.device) * half_size.unsqueeze(0)
        
        # 转换到世界坐标
        corners_expanded = corners.unsqueeze(0).expand(self.num_envs, -1, -1)
        object_rot_expanded = self.climb_object_rot.unsqueeze(1).expand(-1, 8, -1)
        world_corners = quat_rotate(object_rot_expanded.reshape(-1, 4), corners_expanded.reshape(-1, 3))
        world_corners = world_corners.reshape(self.num_envs, 8, 3)
        world_corners += self.climb_object_pos.unsqueeze(1)
        
        return world_corners
        
    def _transform_bbox_to_local(self, bbox_points, heading_rot):
        """将边界框点转换到机器人局部坐标系"""
        rel_points = bbox_points - self.base_pos.unsqueeze(1)
        heading_rot_expanded = heading_rot.unsqueeze(1).expand(-1, 8, -1)
        local_points = quat_rotate(
            heading_rot_expanded.reshape(-1, 4),
            rel_points.reshape(-1, 3)
        )
        return local_points.reshape(self.num_envs, 8, 3)
        
    def _post_physics_step_callback(self):
        """物理步后回调"""
        super()._post_physics_step_callback()
        
        # 更新攀爬进度
        self.climb_progress = (self.base_pos[:, 2] - self.cfg.rewards.base_height_target) / self.object_bbox_size[2]
        self.climb_progress = torch.clamp(self.climb_progress, 0, 1)
        
        # 检查IET条件
        if self.cfg.env.enableIET:
            self._check_iet_condition()
            
    def _check_iet_condition(self):
        """检查交互早期终止条件"""
        # 检查是否到达目标位置
        dist_to_target = torch.norm(self.base_pos - self.climb_target_pos, dim=-1)
        success_mask = dist_to_target < self.cfg.env.successThreshold
        
        # 更新IET计数器
        self.IET_step_buf[success_mask] += 1
        self.IET_step_buf[~success_mask] = 0
        
        # 触发IET
        self.IET_triggered = self.IET_step_buf >= self.cfg.env.maxIETSteps
        
    def check_termination(self):
        """检查终止条件"""
        super().check_termination()
        
        # 添加IET终止
        if self.cfg.env.enableIET:
            self.reset_buf |= self.IET_triggered
            
        # 添加掉落检测
        fell_off = self.base_pos[:, 2] < (self.cfg.rewards.base_height_target - 0.3)
        self.reset_buf |= fell_off
        
    def _reward_climb_progress(self):
        """攀爬进度奖励"""
        height_progress = self.base_pos[:, 2] - self.prev_climb_height
        self.prev_climb_height[:] = self.base_pos[:, 2]
        return torch.clamp(height_progress * 10, -1, 1)
        
    def _reward_feet_clearance(self):
        """脚部离地高度奖励（攀爬时需要抬高脚）"""
        # 在摆动相时奖励脚部抬高
        feet_height = self.feet_pos[:, :, 2] - self.climb_object_pos[:, 2].unsqueeze(1)
        
        # 使用相位信息
        swing_mask = self.leg_phase > 0.5
        clearance_reward = feet_height * swing_mask
        
        return torch.sum(clearance_reward, dim=1) * 0.1
        
    def _reward_climb_stability(self):
        """攀爬稳定性奖励"""
        # 惩罚过大的身体倾斜
        gravity_penalty = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        
        # 惩罚过大的角速度
        ang_vel_penalty = torch.sum(torch.square(self.base_ang_vel), dim=1)
        
        return -gravity_penalty - 0.1 * ang_vel_penalty
        
    def _reward_approach_object(self):
        """接近物体奖励"""
        dist_to_object = torch.norm(
            self.climb_object_pos[:, :2] - self.base_pos[:, :2], 
            dim=-1
        )
        approach_reward = torch.exp(-dist_to_object)
        
        # 只在未开始攀爬时给予此奖励
        not_climbing = self.climb_progress < 0.1
        return approach_reward * not_climbing
        
    def _reward_feet_contact_on_object(self):
        """脚部接触物体奖励"""
        # 检查脚部是否在物体上方
        feet_above_object = self.feet_pos[:, :, 2] > self.climb_object_pos[:, 2].unsqueeze(1)
        
        # 检查脚部是否在物体范围内（简化检查）
        feet_in_range_x = torch.abs(
            self.feet_pos[:, :, 0] - self.climb_object_pos[:, 0].unsqueeze(1)
        ) < self.object_bbox_size[0]/2
        
        feet_in_range_y = torch.abs(
            self.feet_pos[:, :, 1] - self.climb_object_pos[:, 1].unsqueeze(1)
        ) < self.object_bbox_size[1]/2
        
        feet_on_object = feet_above_object & feet_in_range_x & feet_in_range_y
        
        # 结合接触力信息
        feet_contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        
        valid_contact = feet_on_object & feet_contact
        
        return torch.sum(valid_contact.float(), dim=1) * 0.5
        
    def reset_idx(self, env_ids):
        """重置环境"""
        super().reset_idx(env_ids)
        
        if len(env_ids) > 0:
            # 重置攀爬物体位置（随机放置在前方）
            self.climb_object_states[env_ids, 0] = torch_rand_float(
                1.5, 3.0, (len(env_ids),), device=self.device
            )
            self.climb_object_states[env_ids, 1] = torch_rand_float(
                -0.5, 0.5, (len(env_ids),), device=self.device
            )
            self.climb_object_states[env_ids, 2] = 0  # 地面上
            
            # 重置目标位置（物体顶部）
            self.climb_target_pos[env_ids, 0] = self.climb_object_states[env_ids, 0]
            self.climb_target_pos[env_ids, 1] = self.climb_object_states[env_ids, 1]
            self.climb_target_pos[env_ids, 2] = self.object_bbox_size[2] + self.cfg.rewards.base_height_target
            
            # 重置进度
            self.climb_progress[env_ids] = 0
            self.prev_climb_height[env_ids] = self.base_pos[env_ids, 2]
            
            # 重置IET
            if self.cfg.env.enableIET:
                self.IET_step_buf[env_ids] = 0
                self.IET_triggered[env_ids] = False