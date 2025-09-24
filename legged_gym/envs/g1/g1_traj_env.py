import torch
import numpy as np
from legged_gym.envs.g1.g1_env import G1Robot

class G1TrajRobot(G1Robot):
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_trajectory()
        
    def _init_trajectory(self):
        """初始化轨迹生成器"""
        self.traj_points = torch.zeros(
            self.num_envs, 
            self.cfg.traj.num_samples, 
            3,  # x,y,z
            device=self.device
        )
        self._generate_trajectories()
        
    def _generate_trajectories(self):
        """生成随机轨迹"""
        for i in range(self.num_envs):
            # 简单的圆形轨迹示例
            t = torch.linspace(0, 2*np.pi, self.cfg.traj.num_samples, device=self.device)
            radius = torch_rand_float(1.0, 3.0, (1,), device=self.device)
            self.traj_points[i, :, 0] = radius * torch.cos(t) + self.base_pos[i, 0]
            self.traj_points[i, :, 1] = radius * torch.sin(t) + self.base_pos[i, 1]
            self.traj_points[i, :, 2] = self.cfg.rewards.base_height_target
            
    def compute_observations(self):
        super().compute_observations()
        if self.cfg.env.enableTaskObs:
            # 添加轨迹观测
            rel_traj = self.traj_points - self.base_pos.unsqueeze(1)
            traj_obs = rel_traj[:, :5].reshape(self.num_envs, -1)  # 前5个点
            self.obs_buf = torch.cat([self.obs_buf, traj_obs], dim=-1)
            
    def _reward_traj_tracking(self):
        """轨迹跟踪奖励"""
        # 找最近轨迹点
        dists = torch.norm(self.traj_points[:, :, :2] - self.base_pos[:, :2].unsqueeze(1), dim=-1)
        min_dist = torch.min(dists, dim=1)[0]
        return torch.exp(-2.0 * min_dist)
        
    def post_physics_step(self):
        super().post_physics_step()
        # 更新轨迹可视化标记
        if self.cfg.traj.enable_markers and hasattr(self, 'update_markers'):
            self.update_markers(self.traj_points[:, :self.num_markers])