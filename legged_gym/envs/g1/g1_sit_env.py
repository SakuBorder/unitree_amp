import math
import torch

from legged_gym.envs.g1.g1_amp_env import G1AMPRobot
from legged_gym.envs.g1.g1_traj_env import compute_location_observations


def _extract_roll_pitch(quat: torch.Tensor):
    qx = quat[:, 0]
    qy = quat[:, 1]
    qz = quat[:, 2]
    qw = quat[:, 3]

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    return roll, pitch


class G1SitRobot(G1AMPRobot):
    """Task that encourages the G1 humanoid to sit at a target location."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._target_radius = cfg.sit.target_radius
        self._target_radius_min = cfg.sit.min_radius
        self._target_height = cfg.sit.target_height
        self._target_height_noise = cfg.sit.height_noise
        self._pos_err_scale = cfg.sit.pos_error_scale
        self._height_err_scale = cfg.sit.height_error_scale
        self._vel_penalty_scale = cfg.sit.velocity_penalty_scale
        self._ang_vel_penalty_scale = cfg.sit.ang_vel_penalty_scale

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self._sit_targets = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._sit_task_obs = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._sample_sit_targets(env_ids)
        self._update_sit_extras()

    # ------------------------------------------------------------------
    # Noise handling
    # ------------------------------------------------------------------
    def _get_noise_scale_vec(self, cfg):
        base_noise = super()._get_noise_scale_vec(cfg)
        base_obs_dim = cfg.env.num_observations - 3
        if base_obs_dim <= 0:
            raise ValueError("Sit task must reserve proprioceptive observations before task features.")
        if base_noise.shape[-1] < base_obs_dim:
            raise RuntimeError("Base noise vector shorter than proprioceptive observation width.")
        return base_noise[..., :base_obs_dim].clone()

    # ------------------------------------------------------------------
    # Simulation hooks
    # ------------------------------------------------------------------
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if env_ids.numel() > 0:
            self._sample_sit_targets(env_ids)
            self._update_sit_extras()

    def compute_observations(self):
        base_obs = super().compute_observations()
        task_obs = self._compute_task_observation()
        self.obs_buf = torch.cat((base_obs, task_obs), dim=-1)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, task_obs), dim=-1)
        self._update_sit_extras()
        self._refresh_observation_extras()
        return self.obs_buf

    # ------------------------------------------------------------------
    # Task helpers
    # ------------------------------------------------------------------
    def _compute_task_observation(self):
        targets = self._sit_targets.unsqueeze(1)
        local_offsets = compute_location_observations(self.root_states, targets).squeeze(1)
        height_delta = self._sit_targets[:, 2] - self.root_states[:, 2]
        task_obs = torch.stack(
            (local_offsets[:, 0], local_offsets[:, 1], height_delta), dim=-1
        )
        self._sit_task_obs.copy_(task_obs)
        return task_obs

    def _sample_sit_targets(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        radius_span = self._target_radius - self._target_radius_min
        radii = torch.rand(env_ids.shape[0], device=self.device) * radius_span + self._target_radius_min
        angles = torch.rand(env_ids.shape[0], device=self.device) * 2.0 * math.pi
        x = radii * torch.cos(angles)
        y = radii * torch.sin(angles)
        height_noise = (torch.rand(env_ids.shape[0], device=self.device) * 2.0 - 1.0) * self._target_height_noise
        z = torch.full_like(x, self._target_height) + height_noise
        targets = torch.stack((x, y, z), dim=-1)
        self._sit_targets[env_ids] = targets

    def _update_sit_extras(self):
        extras = self.extras.setdefault("sit", {})
        extras["targets"] = self._sit_targets
        extras["task_obs"] = self._sit_task_obs

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _reward_sit_position(self):
        delta = self._sit_targets[:, :2] - self.root_states[:, :2]
        dist_sq = torch.sum(delta * delta, dim=-1)
        return torch.exp(-self._pos_err_scale * dist_sq)

    def _reward_sit_height(self):
        height_delta = self._sit_targets[:, 2] - self.root_states[:, 2]
        return torch.exp(-self._height_err_scale * height_delta * height_delta)

    def _reward_sit_still(self):
        speed_sq = torch.sum(self.base_lin_vel[:, :2] * self.base_lin_vel[:, :2], dim=-1)
        return torch.exp(-self._vel_penalty_scale * speed_sq)

    def _reward_sit_orientation(self):
        root_rot = self.root_states[:, 3:7]
        roll, pitch = _extract_roll_pitch(root_rot)
        ang_err = roll * roll + pitch * pitch
        return torch.exp(-self._ang_vel_penalty_scale * ang_err)