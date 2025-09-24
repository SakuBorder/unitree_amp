import math

import torch
from isaacgym.torch_utils import torch_rand_float

from phc.phc.utils import torch_utils

from legged_gym.envs.g1.g1_env import G1Robot


class G1TrajRobot(G1Robot):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_trajectory_buffers()

    # ------------------------------------------------------------------
    # 轨迹生成
    # ------------------------------------------------------------------
    def _init_trajectory_buffers(self):
        self.traj_points = torch.zeros(
            self.num_envs,
            self.cfg.traj.num_samples,
            3,
            device=self.device,
        )
        self.traj_dirs = torch.zeros(
            self.num_envs,
            self.cfg.traj.num_samples,
            2,
            device=self.device,
        )
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._generate_trajectories(env_ids)

    def _generate_trajectories(self, env_ids):
        if env_ids.numel() == 0:
            return

        dt = self.cfg.traj.sample_timestep
        speed_min = self.cfg.traj.speed_min
        speed_max = self.cfg.traj.speed_max
        accel_max = self.cfg.traj.accel_max
        sharp_turn_prob = self.cfg.traj.sharp_turn_prob
        sharp_turn_angle = self.cfg.traj.sharp_turn_angle

        for env_id in env_ids.tolist():
            start_pos = self.base_pos[env_id, :2]
            points = self.traj_points[env_id]
            dirs = self.traj_dirs[env_id]

            heading = torch_rand_float(-math.pi, math.pi, (1,), device=self.device)[0]
            speed = torch_rand_float(speed_min, speed_max, (1,), device=self.device)[0]

            points[0, 0:2] = start_pos
            points[0, 2] = self.cfg.rewards.base_height_target
            dirs[0] = torch.tensor([math.cos(heading), math.sin(heading)], device=self.device)

            for idx in range(1, self.cfg.traj.num_samples):
                if torch.rand(1, device=self.device) < sharp_turn_prob:
                    heading += torch_rand_float(-sharp_turn_angle, sharp_turn_angle, (1,), device=self.device)[0]
                else:
                    heading += torch_rand_float(-0.25, 0.25, (1,), device=self.device)[0]

                speed_delta = torch_rand_float(-accel_max, accel_max, (1,), device=self.device)[0] * dt
                speed = torch.clamp(speed + speed_delta, speed_min, speed_max)

                dirs[idx] = torch.tensor([math.cos(heading), math.sin(heading)], device=self.device)
                step = dirs[idx] * speed * dt
                points[idx, 0:2] = points[idx - 1, 0:2] + step
                points[idx, 2] = self.cfg.rewards.base_height_target

    # ------------------------------------------------------------------
    # 观测与奖励
    # ------------------------------------------------------------------
    def compute_observations(self):
        super().compute_observations()
        if self.cfg.env.enableTaskObs:
            rel_traj = self.traj_points - self.base_pos.unsqueeze(1)
            num_samples = min(5, self.cfg.traj.num_samples)
            traj_obs = rel_traj[:, :num_samples].reshape(self.num_envs, -1)
            self.obs_buf = torch.cat([self.obs_buf, traj_obs], dim=-1)
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, traj_obs], dim=-1)

    def _reward_traj_tracking(self):
        dists = torch.norm(
            self.traj_points[:, :, :2] - self.base_pos[:, :2].unsqueeze(1),
            dim=-1,
        )
        min_dist = torch.min(dists, dim=1)[0]
        return torch.exp(-2.0 * min_dist)

    def _reward_traj_orientation(self):
        # 取最近轨迹点的方向
        dists = torch.norm(
            self.traj_points[:, :, :2] - self.base_pos[:, :2].unsqueeze(1),
            dim=-1,
        )
        nearest_idx = torch.argmin(dists, dim=1)
        dirs = self.traj_dirs[torch.arange(self.num_envs, device=self.device), nearest_idx]
        target_heading = torch.atan2(dirs[:, 1], dirs[:, 0])
        base_heading = torch_utils.calc_heading_quat(self.base_quat)
        diff = torch.abs(torch_utils.wrap_to_pi(base_heading - target_heading))
        return torch.exp(-diff * 2.0)

    def _reward_smooth_motion(self):
        delta = torch.norm(self.actions - self.last_actions, dim=-1)
        return torch.exp(-5.0 * delta)

    # ------------------------------------------------------------------
    # 状态更新
    # ------------------------------------------------------------------
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        self._generate_trajectories(env_ids)

    def post_physics_step(self):
        super().post_physics_step()
        if self.cfg.traj.enable_markers and hasattr(self, "update_markers"):
            num_markers = min(self.num_markers, self.cfg.traj.num_samples)
            self.update_markers(self.traj_points[:, :num_markers])
