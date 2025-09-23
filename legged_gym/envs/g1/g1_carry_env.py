import torch

from isaacgym.torch_utils import quat_apply

from legged_gym.envs.g1.g1_traj_env import G1TrajRobot


class G1CarryRobot(G1TrajRobot):
    """Trajectory following while maintaining an upright pose as if carrying an object."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._height_target = cfg.carry.height_target
        self._height_noise = cfg.carry.height_noise
        self._height_err_scale = cfg.carry.height_error_scale
        self._ang_vel_penalty_scale = cfg.carry.ang_vel_penalty_scale
        self._lateral_vel_penalty_scale = cfg.carry.lateral_vel_penalty_scale

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self._carry_height_targets = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._world_up = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._world_up[:, 2] = 1.0
        self._carry_task_obs = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._sample_carry_heights(env_ids)
        self._update_carry_extras()

    # ------------------------------------------------------------------
    # Simulation hooks
    # ------------------------------------------------------------------
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if env_ids.numel() > 0:
            self._sample_carry_heights(env_ids)
            self._update_carry_extras()

    def compute_observations(self):
        base_obs = super().compute_observations()
        task_obs = self._compute_task_observation()
        self.obs_buf = torch.cat((base_obs, task_obs), dim=-1)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, task_obs), dim=-1)
        self._update_carry_extras()
        self._refresh_observation_extras()
        return self.obs_buf

    # ------------------------------------------------------------------
    # Task helpers
    # ------------------------------------------------------------------
    def _compute_task_observation(self):
        height_error = (self._carry_height_targets - self.root_states[:, 2]).unsqueeze(-1)
        root_rot = self.root_states[:, 3:7]
        upright = quat_apply(root_rot, self._world_up)[:, 2:3]
        task_obs = torch.cat((height_error, upright), dim=-1)
        self._carry_task_obs.copy_(task_obs)
        return task_obs

    def _sample_carry_heights(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        noise = (torch.rand(env_ids.shape[0], device=self.device) * 2.0 - 1.0) * self._height_noise
        targets = torch.full((env_ids.shape[0],), self._height_target, device=self.device) + noise
        self._carry_height_targets[env_ids] = targets

    def _update_carry_extras(self):
        extras = self.extras.setdefault("carry", {})
        extras["height_targets"] = self._carry_height_targets
        extras["task_obs"] = self._carry_task_obs

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _reward_carry_height(self):
        height_error = self._carry_task_obs[:, 0]
        return torch.exp(-self._height_err_scale * height_error * height_error)

    def _reward_carry_upright(self):
        upright = self._carry_task_obs[:, 1]
        return torch.clamp(upright, min=0.0)

    def _reward_carry_stability(self):
        ang_vel = self.base_ang_vel[:, :2]
        ang_vel_sq = torch.sum(ang_vel * ang_vel, dim=-1)
        return torch.exp(-self._ang_vel_penalty_scale * ang_vel_sq)

    def _reward_carry_lateral_penalty(self):
        lat_vel_sq = torch.sum(self.base_lin_vel[:, :2] * self.base_lin_vel[:, :2], dim=-1)
        return lat_vel_sq * self._lateral_vel_penalty_scale