import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from TokenHSI.tokenhsi.utils import traj_generator
from legged_gym.envs.g1.g1_amp_env import G1AMPRobot


class G1TrajRobot(G1AMPRobot):
    """G1 task with trajectory following observations and rewards (HumanoidTraj-style markers & draw)."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # --- traj config ---
        self._num_traj_samples = cfg.traj.num_samples
        self._traj_sample_timestep = cfg.traj.sample_timestep
        self._traj_num_verts = cfg.traj.num_verts
        self._traj_dtheta_max = cfg.traj.dtheta_max
        self._speed_min = cfg.traj.speed_min
        self._speed_max = cfg.traj.speed_max
        self._accel_max = cfg.traj.accel_max
        self._sharp_turn_prob = cfg.traj.sharp_turn_prob
        self._sharp_turn_angle = cfg.traj.sharp_turn_angle
        self._traj_debug_enabled = cfg.traj.enable_debug_vis

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # cached samples (world/local) and time offsets
        self._traj_samples_world = torch.zeros(self.num_envs, self._num_traj_samples, 3, device=self.device, dtype=torch.float32)
        self._traj_samples_local = torch.zeros(self.num_envs, self._num_traj_samples, 2, device=self.device, dtype=torch.float32)
        self._traj_sample_offsets = (torch.arange(self._num_traj_samples, device=self.device, dtype=torch.float32)
                                     * self._traj_sample_timestep)

        # draw color (pure blue to match HumanoidTraj)
        self._traj_line_color = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        # build generator & prime buffers/extras
        self._build_traj_generator()
        self._refresh_traj_buffers()
        self._update_traj_extras()

        # --- HumanoidTraj-style marker actors (red mini pillars) ---
        self._traj_marker_asset = None
        self._traj_marker_handles = []      # list[list[int]]
        self._traj_marker_actor_ids = None  # Tensor[int32] in SIM domain
        self._load_marker_asset()
        self._create_marker_actors()
        self._build_marker_actor_ids()

    # -------------------- reward cfg --------------------
    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        # 保持与 HumanoidTraj 等价的累计量级：此处不做 dt 缩放修正

    # -------------------- noise -------------------------
    def _get_noise_scale_vec(self, cfg):
        base_noise = super()._get_noise_scale_vec(cfg)
        base_obs_dim = cfg.env.num_observations - 2 * self._num_traj_samples
        if base_obs_dim <= 0:
            raise ValueError("Trajectory configuration must reserve space for proprioceptive observations.")
        if base_noise.shape[-1] == base_obs_dim:
            return base_noise
        if base_noise.shape[-1] < base_obs_dim:
            raise RuntimeError("Base observation noise vector is smaller than the proprioceptive observation width.")
        return base_noise[..., :base_obs_dim].clone()

    # -------------------- resets ------------------------
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if env_ids.numel() > 0:
            self._reset_traj_follow_task(env_ids)
            self._refresh_traj_buffers(env_ids)
            self._update_traj_extras()
            self._update_markers_from_samples()  # keep markers in sync on reset

    # -------------------- observations ------------------
    def compute_observations(self):
        super().compute_observations()
        _, traj_local = self._refresh_traj_buffers()
        traj_obs = traj_local.reshape(self.num_envs, -1)
        self.obs_buf = torch.cat((self.obs_buf, traj_obs), dim=-1)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, traj_obs), dim=-1)
        self._update_traj_extras()
        self._refresh_observation_extras()
        return self.obs_buf

    # -------------------- traj utils --------------------
    def _build_traj_generator(self):
        episode_dur = self.max_episode_length * self.dt
        self._traj_gen = traj_generator.TrajGenerator(
            self.num_envs, episode_dur, self._traj_num_verts, self.device,
            self._traj_dtheta_max, self._speed_min, self._speed_max,
            self._accel_max, self._sharp_turn_prob, self._sharp_turn_angle
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

        t0 = progress.to(dtype=torch.float32) * self.dt
        ts = torch.arange(self._num_traj_samples, device=self.device, dtype=torch.float32) * self._traj_sample_timestep
        traj_ts = t0.unsqueeze(-1) + ts

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_ts.shape)
        pos_flat = self._traj_gen.calc_pos(env_ids_tiled.flatten(), traj_ts.flatten())
        return pos_flat.view(env_ids.shape[0], self._num_traj_samples, -1)

    def _refresh_traj_buffers(self, env_ids=None):
        traj_samples = self._fetch_traj_samples(env_ids)
        root_states = self.root_states if env_ids is None else self.root_states[env_ids]
        traj_local = compute_location_observations(root_states, traj_samples)
        if env_ids is None:
            self._traj_samples_world.copy_(traj_samples)
            self._traj_samples_local.copy_(traj_local)
        else:
            self._traj_samples_world[env_ids] = traj_samples
            self._traj_samples_local[env_ids] = traj_local
        return traj_samples, traj_local

    def _update_traj_extras(self):
        traj_extras = self.extras.setdefault("traj", {})
        traj_extras["samples_world"] = self._traj_samples_world
        traj_extras["samples_local"] = self._traj_samples_local
        if self._traj_samples_world.shape[1] > 0:
            traj_extras["next_target_world"] = self._traj_samples_world[:, 0]
            traj_extras["next_target_local"] = self._traj_samples_local[:, 0]
        traj_extras["sample_offsets"] = self._traj_sample_offsets
        traj_extras["episode_time"] = self.episode_length_buf.to(dtype=torch.float32) * self.dt

    # -------------------- reward ------------------------
    def _reward_traj_tracking(self):
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        time = self.episode_length_buf.to(dtype=torch.float32) * self.dt
        traj_tar_pos = self._traj_gen.calc_pos(env_ids, time)   # 当前时刻 t 的连续目标点
        root_pos = self.root_states[:, 0:3]
        return compute_traj_reward(root_pos, traj_tar_pos)      # 固定系数 2.0，XY 误差

    # -------------------- markers (HumanoidTraj style) --
    def _load_marker_asset(self):
        asset_root = "/home/dy/dy/code/unitree_amp/assert/"
        asset_file = "location_marker.urdf"
        opts = gymapi.AssetOptions()
        opts.angular_damping = 0.01
        opts.linear_damping = 0.01
        opts.max_angular_velocity = 100.0
        opts.density = 1.0
        opts.fix_base_link = True
        opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._traj_marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, opts)

    def _create_marker_actors(self):
        default_pose = gymapi.Transform()
        col_group = self.num_envs + 10
        col_filter = 1
        segmentation_id = 0
        self._traj_marker_handles = []
        for env_ptr in self.envs:
            handles = []
            for _ in range(self._num_traj_samples):
                h = self.gym.create_actor(env_ptr, self._traj_marker_asset, default_pose,
                                          "marker", col_group, col_filter, segmentation_id)
                self.gym.set_actor_scale(env_ptr, h, 0.5)
                self.gym.set_rigid_body_color(env_ptr, h, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.0, 0.0))
                handles.append(h)
            self._traj_marker_handles.append(handles)

    def _build_marker_actor_ids(self):
        ids = []
        for env_idx, env_ptr in enumerate(self.envs):
            for h in self._traj_marker_handles[env_idx]:
                # 修复：移除 self.sim 参数，正确的参数顺序是 (env_ptr, actor_handle, domain)
                aid = self.gym.get_actor_index(env_ptr, h, gymapi.DOMAIN_SIM)
                ids.append(aid)
        self._traj_marker_actor_ids = torch.tensor(ids, device=self.device, dtype=torch.int32)

    def _update_markers_from_samples(self):
        if self._traj_marker_actor_ids is None or self._traj_marker_actor_ids.numel() == 0:
            return
        # 提到当前根高度
        z = self.root_states[:, 2:3]  # [N,1]
        samples = self._traj_samples_world.clone()
        samples[..., 2] = z.repeat(1, self._num_traj_samples)

        num_total = self.num_envs * self._num_traj_samples
        states = torch.zeros((num_total, 13), device=self.device, dtype=torch.float32)
        states[:, 0:3] = samples.reshape(-1, 3)
        states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        # 将 marker 状态写入全局 root state，再按索引下发
        self.root_states[self._traj_marker_actor_ids.long()] = states
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self._traj_marker_actor_ids),
            self._traj_marker_actor_ids.shape[0]
        )

    # -------------------- rendering ---------------------
    def render(self, sync_frame_time=True):
        super().render(sync_frame_time)
        if self.viewer and self._traj_debug_enabled:
            self._refresh_traj_buffers()
            self._update_markers_from_samples()
            self._draw_traj_debug()

    def _draw_traj_debug(self):
        self.gym.clear_lines(self.viewer)
        for env_idx, env_ptr in enumerate(self.envs):
            verts = self._traj_gen.get_traj_verts(env_idx)
            if verts.shape[0] < 2:
                continue
            # 路径高度取当前根高（与 HumanoidTraj 视觉一致）
            h = float(self.root_states[env_idx, 2].item())
            verts = verts.clone()
            verts[..., 2] = h
            lines = torch.cat((verts[:-1], verts[1:]), dim=-1).cpu().numpy()
            cols = np.broadcast_to(self._traj_line_color, (lines.shape[0], self._traj_line_color.shape[-1]))
            self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, cols)


# -------------------- helpers --------------------
def compute_location_observations(root_states, traj_samples):
    num_envs = root_states.shape[0]
    num_samples = traj_samples.shape[1] if traj_samples.dim() >= 2 else 0
    if num_envs == 0:
        return traj_samples.new_zeros((0, num_samples, 2))

    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    sin_yaw = 2.0 * (root_rot[:, 3] * root_rot[:, 2] + root_rot[:, 0] * root_rot[:, 1])
    cos_yaw = 1.0 - 2.0 * (root_rot[:, 1] * root_rot[:, 1] + root_rot[:, 2] * root_rot[:, 2])
    heading = torch.atan2(sin_yaw, cos_yaw)

    cos_h = torch.cos(heading).unsqueeze(1)
    sin_h = torch.sin(heading).unsqueeze(1)

    delta = traj_samples[..., 0:2] - root_pos.unsqueeze(1)[..., 0:2]
    x_local = cos_h * delta[..., 0] + sin_h * delta[..., 1]
    y_local = -sin_h * delta[..., 0] + cos_h * delta[..., 1]

    return torch.stack((x_local, y_local), dim=-1)


@torch.jit.script
def compute_traj_reward(root_pos: torch.Tensor, tar_pos: torch.Tensor) -> torch.Tensor:
    # 与 HumanoidTraj 保持一致：固定系数 2.0，仅 XY 误差
    pos_err_scale = 2.0
    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    return torch.exp(-pos_err_scale * pos_err)