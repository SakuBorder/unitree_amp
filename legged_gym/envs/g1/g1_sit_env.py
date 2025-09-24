import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import json

class G1SitRobot(G1Robot):
    """G1坐下任务环境"""
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 坐下任务需要2个actors: robot + seat_object
        self.actors_per_env = 2
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
    def _create_envs(self):
        """创建环境，包含机器人和座椅"""
        self._load_seat_assets()
        super()._create_envs()
        
    def _load_seat_assets(self):
        """加载座椅资产"""
        self.seat_assets = []
        self.seat_configs = []
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.density = 1000.0
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        
        # 为每种类别创建资产
        for cat in self.cfg.env.objCategories:
            if cat == "chair":
                asset, config = self._create_chair_asset(asset_options)
            elif cat == "stool":
                asset, config = self._create_stool_asset(asset_options)
            elif cat == "bench":
                asset, config = self._create_bench_asset(asset_options)
            else:
                asset, config = self._create_default_seat_asset(asset_options)
                
            self.seat_assets.append(asset)
            self.seat_configs.append(config)
            
    def _create_chair_asset(self, options):
        """创建椅子资产"""
        # 简化版椅子：座位 + 靠背
        seat_width = 0.5
        seat_depth = 0.5
        seat_height = 0.45
        
        asset = self.gym.create_box(
            self.sim, seat_width, seat_depth, seat_height, options
        )
        
        config = {
            "seat_height": seat_height,
            "seat_center": [0, 0, seat_height/2],
            "bbox": [seat_width, seat_depth, seat_height],
            "facing": [1, 0, 0],  # 朝向前方
            "sit_offset": [0, 0, seat_height + 0.1]  # 坐的目标位置
        }
        
        return asset, config
        
    def _create_stool_asset(self, options):
        """创建凳子资产"""
        stool_radius = 0.3
        stool_height = 0.4
        
        # 用箱子简化表示圆凳
        asset = self.gym.create_box(
            self.sim, stool_radius*2, stool_radius*2, stool_height, options
        )
        
        config = {
            "seat_height": stool_height,
            "seat_center": [0, 0, stool_height/2],
            "bbox": [stool_radius*2, stool_radius*2, stool_height],
            "facing": [1, 0, 0],
            "sit_offset": [0, 0, stool_height + 0.1]
        }
        
        return asset, config
        
    def _create_bench_asset(self, options):
        """创建长凳资产"""
        bench_width = 1.2
        bench_depth = 0.4
        bench_height = 0.42
        
        asset = self.gym.create_box(
            self.sim, bench_width, bench_depth, bench_height, options
        )
        
        config = {
            "seat_height": bench_height,
            "seat_center": [0, 0, bench_height/2],
            "bbox": [bench_width, bench_depth, bench_height],
            "facing": [1, 0, 0],
            "sit_offset": [0, 0, bench_height + 0.1]
        }
        
        return asset, config
        
    def _create_default_seat_asset(self, options):
        """创建默认座椅"""
        return self._create_chair_asset(options)
        
    def _build_env(self, env_id, env_ptr, robot_asset):
        """构建单个环境"""
        super()._build_env(env_id, env_ptr, robot_asset)
        
        # 选择座椅类型
        seat_id = env_id % len(self.seat_assets)
        seat_asset = self.seat_assets[seat_id]
        seat_config = self.seat_configs[seat_id]
        
        # 放置座椅
        seat_pose = gymapi.Transform()
        seat_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)  # 前方1米
        
        # 随机旋转
        if self.cfg.env.mode == "train":
            yaw = np.random.uniform(-np.pi/4, np.pi/4)
        else:
            yaw = 0
        seat_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), yaw)
        
        seat_handle = self.gym.create_actor(
            env_ptr, seat_asset, seat_pose,
            f"seat_{env_id}", env_id, 0, 0
        )
        
        # 设置颜色
        self.gym.set_rigid_body_color(
            env_ptr, seat_handle, 0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(0.6, 0.4, 0.2)  # 棕色
        )
        
        # 保存配置
        if not hasattr(self, 'env_seat_configs'):
            self.env_seat_configs = []
        self.env_seat_configs.append(seat_config)
        
    def _init_buffers(self):
        """初始化缓冲区"""
        super()._init_buffers()
        
        # 座椅状态
        self.seat_states = self.all_root_states[:, 1]  # [num_envs, 13]
        self.seat_pos = self.seat_states[:, 0:3]
        self.seat_rot = self.seat_states[:, 3:7]
        
        # 目标坐姿位置
        self.sit_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.sit_target_orient = torch.zeros(self.num_envs, 4, device=self.device)
        self.sit_target_orient[:, 3] = 1  # 单位四元数
        
        # 座椅配置转tensor
        self._setup_seat_configs()
        
        # 坐下状态跟踪
        self.is_sitting = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.sit_stability_buf = torch.zeros(self.num_envs, device=self.device)
        self.prev_base_vel = torch.zeros(self.num_envs, 6, device=self.device)
        
    def _setup_seat_configs(self):
        """设置座椅配置张量"""
        configs = self.env_seat_configs if hasattr(self, 'env_seat_configs') else [self.seat_configs[0]] * self.num_envs
        
        self.seat_heights = torch.tensor(
            [c["seat_height"] for c in configs],
            device=self.device
        )
        
        self.seat_bbox = torch.tensor(
            [c["bbox"] for c in configs],
            device=self.device
        )
        
        self.seat_facing = torch.tensor(
            [c["facing"] for c in configs],
            device=self.device
        )
        
        self.seat_sit_offset = torch.tensor(
            [c["sit_offset"] for c in configs],
            device=self.device
        )
        
    def compute_observations(self):
        """计算观测"""
        super().compute_observations()
        
        if self.cfg.env.enableTaskObs:
            task_obs = self._compute_task_obs()
            self.obs_buf = torch.cat([self.obs_buf, task_obs], dim=-1)
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, task_obs], dim=-1)
                
    def _compute_task_obs(self):
        """计算任务观测"""
        # 计算相对位置
        rel_seat_pos = self.seat_pos - self.base_pos
        rel_sit_target = self.sit_target_pos - self.base_pos
        
        # 转换到机器人局部坐标系
        heading_rot = torch_utils.calc_heading_quat_inv(self.base_quat)
        local_seat_pos = quat_rotate(heading_rot, rel_seat_pos)
        local_sit_target = quat_rotate(heading_rot, rel_sit_target)
        
        # 座椅朝向
        local_seat_rot = quat_mul(heading_rot, self.seat_rot)
        seat_rot_obs = torch_utils.quat_to_tan_norm(local_seat_rot)
        
        # 座椅朝向向量
        seat_facing_world = quat_rotate(self.seat_rot, self.seat_facing)
        local_seat_facing = quat_rotate(heading_rot, seat_facing_world)
        
        # 座椅边界框（8个角点）
        bbox_points = self._compute_seat_bbox_points()
        local_bbox = self._transform_to_local(bbox_points, heading_rot)
        
        # 坐姿稳定性指标
        stability = self.sit_stability_buf.unsqueeze(-1)
        
        task_obs = torch.cat([
            local_sit_target,                              # 3
            local_seat_pos,                                # 3
            seat_rot_obs,                                  # 6
            local_seat_facing[:, :2],                      # 2（只要水平朝向）
            local_bbox.reshape(self.num_envs, -1)[:, :24], # 24
            stability                                       # 1
        ], dim=-1)
        
        return task_obs
        
    def _compute_seat_bbox_points(self):
        """计算座椅边界框8个角点"""
        half_size = self.seat_bbox / 2
        
        corners = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ], device=self.device, dtype=torch.float)
        
        corners = corners.unsqueeze(0) * half_size.unsqueeze(1)
        
        # 旋转到世界坐标
        seat_rot_expanded = self.seat_rot.unsqueeze(1).expand(-1, 8, -1)
        world_corners = quat_rotate(
            seat_rot_expanded.reshape(-1, 4),
            corners.reshape(-1, 3)
        )
        world_corners = world_corners.reshape(self.num_envs, 8, 3)
        world_corners += self.seat_pos.unsqueeze(1)
        
        return world_corners
        
    def _transform_to_local(self, points, heading_rot):
        """转换点到局部坐标系"""
        rel_points = points - self.base_pos.unsqueeze(1)
        heading_rot_expanded = heading_rot.unsqueeze(1).expand(-1, points.shape[1], -1)
        local_points = quat_rotate(
            heading_rot_expanded.reshape(-1, 4),
            rel_points.reshape(-1, 3)
        )
        return local_points.reshape(self.num_envs, -1, 3)
        
    def _post_physics_step_callback(self):
        """物理步后处理"""
        super()._post_physics_step_callback()
        
        # 更新目标坐姿位置
        self._update_sit_targets()
        
        # 检查坐姿状态
        self._check_sitting_state()
        
        # 更新稳定性
        self._update_stability()
        
    def _update_sit_targets(self):
        """更新目标坐姿位置"""
        # 目标位置 = 座椅位置 + 坐姿偏移
        seat_rot_expanded = self.seat_rot
        sit_offset_world = quat_rotate(seat_rot_expanded, self.seat_sit_offset)
        self.sit_target_pos = self.seat_pos + sit_offset_world
        
        # 目标朝向应该与座椅朝向一致
        self.sit_target_orient = self.seat_rot
        
