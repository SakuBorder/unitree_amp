import torch
from isaacgym import gymapi, gymtorch
from legged_gym.envs.g1.g1_env import G1Robot

class G1CarryRobot(G1Robot):
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 搬运任务需要2个actors: robot + box
        self.actors_per_env = 2
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
    def _create_envs(self):
        super()._create_envs()
        self._create_boxes()
        
    def _create_boxes(self):
        """创建箱子资产"""
        self.box_handles = []
        asset_options = gymapi.AssetOptions()
        asset_options.density = 100.0 if not self.cfg.box.build.randomDensity else 500.0
        
        for i in range(self.num_envs):
            box_size = torch.tensor(self.cfg.box.build.baseSize) 
            box_asset = self.gym.create_box(
                self.sim, 
                box_size[0], box_size[1], box_size[2],
                asset_options
            )
            
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(0.5, 0.0, box_size[2]/2)
            
            box_handle = self.gym.create_actor(
                self.envs[i], box_asset, start_pose,
                "box", i, 0, 0
            )
            self.box_handles.append(box_handle)
            
    def _init_buffers(self):
        super()._init_buffers()
        
        # 箱子状态
        self.box_states = self.all_root_states[:, 1]  # [num_envs, 13]
        self.box_pos = self.box_states[:, 0:3]
        self.box_rot = self.box_states[:, 3:7]
        self.box_vel = self.box_states[:, 7:10]
        
        # 目标位置
        self.tar_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_box_pos = torch.zeros_like(self.box_pos)
        
        # 手部关键点
        self.hand_indices = [5, 11]  # 左右踝关节作为抓取点
        
    def compute_observations(self):
        super().compute_observations()
        if self.cfg.env.enableTaskObs:
            task_obs = self._compute_task_obs()
            self.obs_buf = torch.cat([self.obs_buf, task_obs], dim=-1)
            
    def _compute_task_obs(self):
        """计算任务观测"""
        # 相对位置和速度
        rel_box_pos = self.box_pos - self.base_pos
        rel_tar_pos = self.tar_pos - self.base_pos
        
        # 箱子朝向
        box_rot_obs = torch.cat([
            self.box_rot[:, -1:],  # w
            self.box_rot[:, :3]    # x,y,z
        ], dim=-1)
        
        return torch.cat([
            rel_tar_pos,
            rel_box_pos,
            box_rot_obs,
            self.box_vel,
        ], dim=-1)
        
    def _reward_carry_walk(self):
        """靠近箱子的奖励"""
        dist = torch.norm(self.box_pos[:, :2] - self.base_pos[:, :2], dim=-1)
        return torch.exp(-0.5 * dist)
        
    def _reward_carry_vel(self):
        """搬运速度奖励"""
        box_to_tar = self.tar_pos[:, :2] - self.box_pos[:, :2]
        box_to_tar_dir = box_to_tar / (torch.norm(box_to_tar, dim=-1, keepdim=True) + 1e-6)
        box_vel_2d = self.box_vel[:, :2]
        vel_reward = torch.sum(box_to_tar_dir * box_vel_2d, dim=-1)
        return torch.exp(-2.0 * (1.5 - vel_reward)**2)
        
    def _reward_handheld(self):
        """抓取奖励"""
        hand_pos = self.feet_pos[:, :2]  # 使用脚部位置模拟手
        hand_to_box = torch.norm(hand_pos.mean(dim=1) - self.box_pos[:, :2], dim=-1)
        return torch.exp(-5.0 * hand_to_box)
        
    def _reward_putdown(self):
        """放置奖励"""
        at_target = torch.norm(self.box_pos - self.tar_pos, dim=-1) < 0.1
        on_ground = torch.abs(self.box_pos[:, 2] - self.cfg.box.build.baseSize[2]/2) < 0.01
        return (at_target & on_ground).float()
        
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            # 重置箱子位置
            self.box_states[env_ids, 0:2] = torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
            self.box_states[env_ids, 2] = self.cfg.box.build.baseSize[2]/2
            self.box_states[env_ids, 3:7] = torch.tensor([0, 0, 0, 1], device=self.device)
            self.box_states[env_ids, 7:] = 0
            
            # 重置目标位置
            self.tar_pos[env_ids, 0:2] = torch_rand_float(-2, 2, (len(env_ids), 2), device=self.device)
            self.tar_pos[env_ids, 2] = self.cfg.box.build.baseSize[2]/2