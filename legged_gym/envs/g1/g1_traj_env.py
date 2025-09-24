import math

import torch
from isaacgym.torch_utils import quat_rotate

from phc.phc.utils import torch_utils

from legged_gym.envs.g1.g1_amp_env import G1AMPRobot


def _compute_location_observations(root_states: torch.Tensor, traj_samples: torch.Tensor) -> torch.Tensor:
    """Replicates TokenHSI's trajectory observation builder.

    The samples are expressed in the world frame. This function projects the
    XY offsets into the robot heading frame and flattens them.
    """

    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    heading_rot_exp = heading_rot.unsqueeze(1).expand(-1, traj_samples.shape[1], -1)
    root_pos_exp = root_pos.unsqueeze(1).expand_as(traj_samples)

    local_samples = quat_rotate(
        heading_rot_exp.reshape(-1, 4),
        (traj_samples.reshape(-1, 3) - root_pos_exp.reshape(-1, 3)),
    ).reshape(root_pos.shape[0], traj_samples.shape[1], 3)

    return local_samples[..., 0:2].reshape(root_pos.shape[0], -1)


class G1TrajRobot(G1AMPRobot):

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

            heading = float((torch.rand(1, device=self.device) * 2.0 - 1.0) * math.pi)
            speed = float(torch.rand(1, device=self.device) * (speed_max - speed_min) + speed_min)

            points[0, 0:2] = start_pos
            points[0, 2] = self.cfg.rewards.base_height_target
            dirs[0] = torch.tensor([math.cos(heading), math.sin(heading)], device=self.device, dtype=torch.float32)

            for idx in range(1, self.cfg.traj.num_samples):
                delta_heading = float(torch.rand(1, device=self.device))
                if delta_heading < sharp_turn_prob:
                    delta_heading = float((torch.rand(1, device=self.device) * 2.0 - 1.0) * sharp_turn_angle)
                else:
                    delta_heading = float((torch.rand(1, device=self.device) * 0.5 - 0.25))
                heading += delta_heading

                speed_delta = float((torch.rand(1, device=self.device) * 2.0 - 1.0) * accel_max * dt)
                speed = max(min(speed + speed_delta, speed_max), speed_min)

                dirs[idx] = torch.tensor([math.cos(heading), math.sin(heading)], device=self.device, dtype=torch.float32)
                step = dirs[idx] * speed * dt
                points[idx, 0:2] = points[idx - 1, 0:2] + step
                points[idx, 2] = self.cfg.rewards.base_height_target

    # ------------------------------------------------------------------
    # 观测与奖励
    # ------------------------------------------------------------------
    def compute_observations(self):
        super().compute_observations()
        if self.cfg.env.enableTaskObs:
            num_samples = min(self.cfg.traj.num_samples, getattr(self.cfg.traj, "num_obs_samples", self.cfg.traj.num_samples))
            traj_samples = self.traj_points[:, :num_samples]
            traj_obs = _compute_location_observations(self.root_states, traj_samples)
            self.obs_buf = torch.cat([self.obs_buf, traj_obs], dim=-1)
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, traj_obs], dim=-1)
        self._refresh_observation_extras()

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
