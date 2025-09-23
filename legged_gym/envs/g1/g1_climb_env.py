import math
import torch

from isaacgym.torch_utils import quat_apply

from legged_gym.envs.g1.g1_amp_env import G1AMPRobot
from legged_gym.envs.g1.g1_traj_env import compute_location_observations


class G1ClimbRobot(G1AMPRobot):
    """Task that asks the humanoid to reach an elevated target."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._dist_min, self._dist_max = cfg.climb.distance_range
        self._lateral_range = cfg.climb.lateral_range
        self._height_min, self._height_max = cfg.climb.height_range
        self._height_err_scale = cfg.climb.height_error_scale

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self._climb_targets = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._climb_task_obs = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._climb_prev_planar = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._climb_progress_delta = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._world_up = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._world_up[:, 2] = 1.0

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._sample_climb_targets(env_ids)
        self._reset_climb_state(env_ids)
        self._update_climb_extras()

    # ------------------------------------------------------------------
    # Noise handling
    # ------------------------------------------------------------------
    def _get_noise_scale_vec(self, cfg):
        base_noise = super()._get_noise_scale_vec(cfg)
        base_obs_dim = cfg.env.num_observations - 3
        if base_obs_dim <= 0:
            raise ValueError("Climb task must reserve proprioceptive observations before task features.")
        if base_noise.shape[-1] < base_obs_dim:
            raise RuntimeError("Base noise vector shorter than proprioceptive observation width.")
        return base_noise[..., :base_obs_dim].clone()

    # ------------------------------------------------------------------
    # Simulation hooks
    # ------------------------------------------------------------------
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if env_ids.numel() > 0:
            self._sample_climb_targets(env_ids)
            self._reset_climb_state(env_ids)
            self._update_climb_extras()

    def compute_observations(self):
        base_obs = super().compute_observations()
        task_obs = self._compute_task_observation()
        self.obs_buf = torch.cat((base_obs, task_obs), dim=-1)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, task_obs), dim=-1)
        self._update_climb_extras()
        self._refresh_observation_extras()
        return self.obs_buf

    # ------------------------------------------------------------------
    # Task helpers
    # ------------------------------------------------------------------
    def _compute_task_observation(self):
        targets = self._climb_targets.unsqueeze(1)
        local_offsets = compute_location_observations(self.root_states, targets).squeeze(1)
        height_delta = self._climb_targets[:, 2] - self.root_states[:, 2]
        task_obs = torch.stack(
            (local_offsets[:, 0], local_offsets[:, 1], height_delta), dim=-1
        )
        self._climb_task_obs.copy_(task_obs)

        planar_dist = torch.norm(local_offsets[:, :2], dim=-1)
        progress_delta = torch.clamp(self._climb_prev_planar - planar_dist, min=0.0)
        self._climb_progress_delta.copy_(progress_delta)
        self._climb_prev_planar.copy_(planar_dist)
        return task_obs

    def _sample_climb_targets(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        distances = torch.rand(env_ids.shape[0], device=self.device) * (self._dist_max - self._dist_min) + self._dist_min
        angles = torch.rand(env_ids.shape[0], device=self.device) * 2.0 * math.pi
        x = distances * torch.cos(angles)
        y = torch.clamp(distances * torch.sin(angles), -self._lateral_range, self._lateral_range)
        heights = torch.rand(env_ids.shape[0], device=self.device) * (self._height_max - self._height_min) + self._height_min
        targets = torch.stack((x, y, heights), dim=-1)
        self._climb_targets[env_ids] = targets

    def _reset_climb_state(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        root_pos = self.root_states[env_ids, 0:3]
        targets = self._climb_targets[env_ids]
        delta = targets[:, :2] - root_pos[:, :2]
        planar = torch.norm(delta, dim=-1)
        self._climb_prev_planar[env_ids] = planar
        self._climb_progress_delta[env_ids] = 0.0

    def _update_climb_extras(self):
        extras = self.extras.setdefault("climb", {})
        extras["targets"] = self._climb_targets
        extras["task_obs"] = self._climb_task_obs
        extras["progress"] = self._climb_progress_delta

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _reward_climb_progress(self):
        return self._climb_progress_delta

    def _reward_climb_height(self):
        height_delta = self._climb_task_obs[:, 2]
        return torch.exp(-self._height_err_scale * height_delta * height_delta)

    def _reward_climb_upright(self):
        root_rot = self.root_states[:, 3:7]
        upright = quat_apply(root_rot, self._world_up)[:, 2]
        return torch.clamp(upright, min=0.0)

    def _reward_climb_slip(self):
        lateral_vel = torch.norm(self.feet_vel[:, :, :2], dim=-1)
        return torch.mean(lateral_vel, dim=-1)
