import os
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from TokenHSI.tokenhsi.utils import traj_generator, torch_utils
from legged_gym.envs.g1.g1_amp_env import G1AMPRobot
from legged_gym import LEGGED_GYM_ROOT_DIR


class G1TrajRobot(G1AMPRobot):
    """G1 轨迹跟随任务（仅轨迹采样/观测/奖励；不创建 marker actor，支持线段调试绘制）。"""

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

        # 缓存的轨迹采样（world/local）与时间偏移
        self._traj_samples_world = torch.zeros(
            self.num_envs, self._num_traj_samples, 3, device=self.device, dtype=torch.float32
        )
        self._traj_samples_local = torch.zeros(
            self.num_envs, self._num_traj_samples, 2, device=self.device, dtype=torch.float32
        )
        self._traj_sample_offsets = (
            torch.arange(self._num_traj_samples, device=self.device, dtype=torch.float32) * self._traj_sample_timestep
        )

        # 线段绘制颜色（可选 debug）
        self._traj_line_color = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        # 构建轨迹生成器并初始化缓冲
        self._build_traj_generator()
        self._refresh_traj_buffers()
        self._update_traj_extras()

    # -------------------- reward cfg --------------------
    def _prepare_reward_function(self):
        super()._prepare_reward_function()

    # -------------------- noise -------------------------
    def _get_noise_scale_vec(self, cfg):
        """Return a noise vector that only spans the proprioceptive observations.

        ``LeggedRobot`` expects ``self.noise_scale_vec`` to match the width of the
        observation tensor that is assembled inside ``G1Robot.compute_observations``.
        For the trajectory task we append additional features *after* that base call,
        so only the proprioceptive slice should receive stochastic perturbations.

        If we were to return the full trajectory-extended length here (as the parent
        implementation does by default) the later noise application would attempt to
        add tensors of mismatched widths, triggering the runtime error observed
        during initialisation.
        """

        base_noise = super()._get_noise_scale_vec(cfg)
        base_obs_dim = cfg.env.num_observations - 2 * cfg.traj.num_samples
        if base_obs_dim <= 0:
            raise ValueError(
                "Trajectory task must reserve proprioceptive observations before task features."
            )
        if base_noise.shape[-1] < base_obs_dim:
            raise RuntimeError(
                "Base noise vector shorter than proprioceptive observation width."
            )
        return base_noise[..., :base_obs_dim].clone()

    # -------------------- resets ------------------------
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if torch.is_tensor(env_ids) and env_ids.numel() > 0:
            self._reset_traj_follow_task(env_ids)
            self._refresh_traj_buffers(env_ids)
            self._update_traj_extras()

    # -------------------- observations ------------------
    def compute_observations(self):
        # 先让基类生成本体观测并加噪
        super().compute_observations()
        # 然后拼接轨迹局部坐标（每样本 2 维）
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
            self.num_envs,
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

    # -------------------- rendering（仅画线，完全不触碰 markers） ----
    def render(self, sync_frame_time=True):
        super().render(sync_frame_time)
        if self.viewer and self._traj_debug_enabled:
            self._refresh_traj_buffers()
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
    if num_envs == 0 or num_samples == 0:
        return traj_samples.new_zeros((0, num_samples, 2))

    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(1), (num_envs, num_samples, 4))
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (num_envs, num_samples, 3))

    local_traj_samples = torch_utils.quat_rotate(
        heading_rot_exp.reshape(-1, 4),
        traj_samples.reshape(-1, 3) - root_pos_exp.reshape(-1, 3)
    )

    return local_traj_samples[..., 0:2].reshape(num_envs, num_samples, 2)


@torch.jit.script
def compute_traj_reward(root_pos: torch.Tensor, tar_pos: torch.Tensor) -> torch.Tensor:
    # 与 HumanoidTraj 保持一致：固定系数 2.0，仅 XY 误差
    pos_err_scale = 2.0
    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    return torch.exp(-pos_err_scale * pos_err)
