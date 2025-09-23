import torch

from TokenHSI.tokenhsi.utils import traj_generator

from legged_gym.envs.g1.g1_amp_env import G1AMPRobot


class G1TrajRobot(G1AMPRobot):
    """G1 task with trajectory following observations and rewards."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._num_traj_samples = cfg.traj.num_samples
        self._traj_sample_timestep = cfg.traj.sample_timestep
        self._traj_num_verts = cfg.traj.num_verts
        self._traj_dtheta_max = cfg.traj.dtheta_max
        self._speed_min = cfg.traj.speed_min
        self._speed_max = cfg.traj.speed_max
        self._accel_max = cfg.traj.accel_max
        self._sharp_turn_prob = cfg.traj.sharp_turn_prob
        self._sharp_turn_angle = cfg.traj.sharp_turn_angle
        self._traj_pos_err_scale = cfg.traj.pos_error_scale

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self._build_traj_generator()

    # ------------------------------------------------------------------
    # Noise handling
    # ------------------------------------------------------------------
    def _get_noise_scale_vec(self, cfg):
        """Restrict stochastic observation noise to the proprioceptive slice."""

        base_noise = super()._get_noise_scale_vec(cfg)
        base_obs_dim = cfg.env.num_observations - 2 * self._num_traj_samples

        if base_obs_dim <= 0:
            raise ValueError("Trajectory configuration must reserve space for proprioceptive observations.")

        if base_noise.shape[-1] == base_obs_dim:
            return base_noise

        if base_noise.shape[-1] < base_obs_dim:
            raise RuntimeError(
                "Base observation noise vector is smaller than the proprioceptive observation width."
            )

        return base_noise[..., :base_obs_dim].clone()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if env_ids.numel() > 0:
            self._reset_traj_follow_task(env_ids)

    def compute_observations(self):
        super().compute_observations()
        traj_obs = self._compute_traj_obs()
        self.obs_buf = torch.cat((self.obs_buf, traj_obs), dim=-1)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, traj_obs), dim=-1)
        self._refresh_observation_extras()
        return self.obs_buf

    def _compute_traj_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self.root_states
        else:
            root_states = self.root_states[env_ids]
        traj_samples = self._fetch_traj_samples(env_ids)
        return compute_location_observations(root_states, traj_samples)

    def _build_traj_generator(self):
        num_envs = self.num_envs
        episode_dur = self.max_episode_length * self.dt
        self._traj_gen = traj_generator.TrajGenerator(
            num_envs,
            episode_dur,
            self._traj_num_verts,
            self.device,
            self._traj_dtheta_max,
            self._speed_min,
            self._speed_max,
            self._accel_max,
            self._sharp_turn_prob,
            self._sharp_turn_angle,
        )
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        root_pos = self.root_states[env_ids, 0:3]
        self._traj_gen.reset(env_ids, root_pos)

    def _reset_traj_follow_task(self, env_ids):
        if env_ids.numel() == 0:
            return
        root_pos = self.root_states[env_ids, 0:3]
        self._traj_gen.reset(env_ids, root_pos)

    def _fetch_traj_samples(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            progress = self.episode_length_buf
        else:
            progress = self.episode_length_buf[env_ids]
        if env_ids.numel() == 0:
            return torch.zeros((0, self._num_traj_samples, 3), device=self.device, dtype=torch.float32)

        timestep_beg = progress.to(dtype=torch.float32) * self.dt
        timesteps = torch.arange(self._num_traj_samples, device=self.device, dtype=torch.float32)
        timesteps = timesteps * self._traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)
        traj_samples_flat = self._traj_gen.calc_pos(env_ids_tiled.flatten(), traj_timesteps.flatten())
        traj_samples = traj_samples_flat.view(env_ids.shape[0], self._num_traj_samples, -1)
        return traj_samples

    def _reward_traj_tracking(self):
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        time = self.episode_length_buf.to(dtype=torch.float32) * self.dt
        traj_tar_pos = self._traj_gen.calc_pos(env_ids, time)
        root_pos = self.root_states[:, 0:3]
        return compute_traj_reward(root_pos, traj_tar_pos, self._traj_pos_err_scale)


def compute_location_observations(root_states, traj_samples):
    if root_states.shape[0] == 0:
        return torch.zeros((0, traj_samples.shape[1] * 2), device=traj_samples.device, dtype=traj_samples.dtype)

    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    sin_yaw = 2.0 * (root_rot[:, 3] * root_rot[:, 2] + root_rot[:, 0] * root_rot[:, 1])
    cos_yaw = 1.0 - 2.0 * (root_rot[:, 1] * root_rot[:, 1] + root_rot[:, 2] * root_rot[:, 2])
    heading = torch.atan2(sin_yaw, cos_yaw)

    cos_heading = torch.cos(heading)
    sin_heading = torch.sin(heading)

    delta = traj_samples[..., 0:2] - root_pos.unsqueeze(1)[..., 0:2]
    x_local = cos_heading.unsqueeze(1) * delta[..., 0] + sin_heading.unsqueeze(1) * delta[..., 1]
    y_local = -sin_heading.unsqueeze(1) * delta[..., 0] + cos_heading.unsqueeze(1) * delta[..., 1]

    obs = torch.stack((x_local, y_local), dim=-1).reshape(root_states.shape[0], -1)
    return obs


def compute_traj_reward(root_pos, tar_pos, pos_err_scale):
    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    return torch.exp(-pos_err_scale * pos_err)
