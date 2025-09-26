import math
import torch
from isaacgym.torch_utils import quat_rotate
from phc.phc.utils import torch_utils
from legged_gym.envs.g1.g1_amp_env import G1AMPRobot
from legged_gym.utils.math import wrap_to_pi
from TokenHSI.tokenhsi.utils.traj_generator import TrajGenerator


# 世界->heading局部系，取XY并展平（与 HumanoidTraj 一致）
def _compute_location_observations(root_states: torch.Tensor, traj_samples: torch.Tensor) -> torch.Tensor:
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)                       # [N,4]

    heading_rot_exp = heading_rot.unsqueeze(1).expand(-1, traj_samples.shape[1], -1)
    root_pos_exp    = root_pos.unsqueeze(1).expand_as(traj_samples)

    local_samples = quat_rotate(
        heading_rot_exp.reshape(-1, 4),
        (traj_samples.reshape(-1, 3) - root_pos_exp.reshape(-1, 3)),
    ).reshape(root_pos.shape[0], traj_samples.shape[1], 3)

    return local_samples[..., 0:2].reshape(root_pos.shape[0], -1)                   # [N,2S]


def _quat_to_yaw_xyzw(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(t0, t1)


class G1TrajRobot(G1AMPRobot):
    """
    轨迹跟随 + AMP（与 HumanoidTraj 对齐），不重写 _compute_rewards：
    - 轨迹：固定 101 顶点（≈0.1s 分段）；按 episode_dur 生成整条轨迹
    - 观测：TokenHSI 风格的局部XY轨迹窗口
    - 奖励分项：_reward_traj_tracking / _reward_traj_orientation / _reward_smooth_motion
      由父类按 cfg.rewards.scales 聚合
    - 重置：追加“偏离轨迹过远”失败
    """

    # ----------------- 基础 -----------------
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_trajectory_generator()

        # 偏离失败阈值（与参考实现一致）
        self._fail_dist = getattr(self.cfg.traj, "fail_dist", 4.0)

        # 评估（可选）
        self._is_eval = bool(getattr(cfg, "eval", False))
        if self._is_eval:
            self._success_threshold = float(getattr(self.cfg.traj, "success_threshold", 1.0))
            self._success_buf   = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self._precision_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def _curr_time(self) -> torch.Tensor:
        # 与 HumanoidTraj 保持一致：都用 progress_buf * dt
        return getattr(self, "progress_buf", torch.zeros(self.num_envs, device=self.device)).float() * self.dt

    # ----------------- 轨迹生成（对齐 HumanoidTraj） -----------------
    def _init_trajectory_generator(self):
        episode_dur = self.max_episode_length * self.dt
        num_verts   = 101          # 100 段 ≈0.1s/段
        dtheta_max  = 2.0

        self._traj_gen = TrajGenerator(
            num_envs     = self.num_envs,
            episode_dur  = episode_dur,
            num_verts    = num_verts,
            device       = self.device,
            dtheta_max   = dtheta_max,
            speed_min    = self.cfg.traj.speed_min,
            speed_max    = self.cfg.traj.speed_max,
            accel_max    = self.cfg.traj.accel_max,
            sharp_turn_prob  = self.cfg.traj.sharp_turn_prob,
            sharp_turn_angle = getattr(self.cfg.traj, "sharp_turn_angle", math.pi),
        )

        # 预计算可视化相关参数，便于随时间调整 marker 窗口
        self._traj_duration    = episode_dur
        self._traj_vert_dt     = episode_dur / float(num_verts - 1)
        self._marker_speed_min = max(0.1, float(getattr(self.cfg.traj, "speed_min", 0.1)))
        window = getattr(self.cfg.traj, "sample_timestep", 0.5) * max(1, getattr(self.cfg.traj, "num_samples", 1) - 1)
        self._marker_max_lag   = max(window, 0.0)

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        init_pos = self.base_pos.clone()
        init_pos[:, 2] = self.cfg.rewards.base_height_target
        self._traj_gen.reset(env_ids, init_pos)

    def _regenerate_trajectories(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        init_pos = self.base_pos[env_ids].clone()
        init_pos[:, 2] = self.cfg.rewards.base_height_target
        self._traj_gen.reset(env_ids, init_pos)

    # ----------------- 观测（滚动窗口） -----------------
    def _fetch_traj_samples(self, S: int = None, env_ids=None) -> torch.Tensor:
        if S is None:
            S = self.cfg.traj.num_samples
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        t_beg = self.progress_buf[env_ids] * self.dt                                 # [n]
        steps = torch.arange(S, device=self.device, dtype=torch.float) * self.cfg.traj.sample_timestep
        ts    = t_beg.unsqueeze(-1) + steps                                          # [n,S]

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), ts.shape)
        samples_flat  = self._traj_gen.calc_pos(env_ids_tiled.reshape(-1), ts.reshape(-1))
        return samples_flat.reshape(env_ids.shape[0], S, -1)                          # [n,S,3]

    def _fetch_marker_samples(self, S: int, env_ids=None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0 or S <= 0:
            return torch.zeros((env_ids.numel(), max(S, 0), 3), device=self.device, dtype=torch.float)

        nominal_t = self.progress_buf[env_ids].float() * self.dt
        start_t   = self._lag_adjusted_start_time(env_ids, nominal_t)

        steps = torch.arange(S, device=self.device, dtype=torch.float) * self.cfg.traj.sample_timestep
        ts    = start_t.unsqueeze(-1) + steps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), ts.shape)
        samples_flat  = self._traj_gen.calc_pos(env_ids_tiled.reshape(-1), ts.reshape(-1))
        return samples_flat.reshape(env_ids.shape[0], S, -1)

    def _lag_adjusted_start_time(self, env_ids: torch.Tensor, nominal_t: torch.Tensor) -> torch.Tensor:
        if env_ids.numel() == 0:
            return nominal_t

        traj_pos = self._traj_gen.calc_pos(env_ids, nominal_t)
        robot_xy = self.base_pos[env_ids, :2]
        err_xy   = torch.norm(traj_pos[:, :2] - robot_xy, dim=-1)

        next_t   = torch.clamp(nominal_t + self._traj_vert_dt, max=self._traj_duration)
        next_pos = self._traj_gen.calc_pos(env_ids, next_t)
        delta_t  = torch.clamp(next_t - nominal_t, min=1e-6)
        exp_speed = torch.norm(next_pos[:, :2] - traj_pos[:, :2], dim=-1) / delta_t
        exp_speed = torch.clamp(exp_speed, min=self._marker_speed_min)

        lag = err_xy / exp_speed
        if self._marker_max_lag > 0:
            lag = torch.clamp(lag, max=self._marker_max_lag)

        return torch.clamp(nominal_t - lag, min=0.0)

    def compute_observations(self):
        super().compute_observations()
        if getattr(self.cfg.env, "enableTaskObs", False):
            S = min(self.cfg.traj.num_samples,
                    getattr(self.cfg.traj, "num_obs_samples", self.cfg.traj.num_samples))
            samples  = self._fetch_traj_samples(S)
            traj_obs = _compute_location_observations(self.root_states, samples)     # [N,2S]
            self.obs_buf = torch.cat([self.obs_buf, traj_obs], dim=-1)
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, traj_obs], dim=-1)
        self._refresh_observation_extras()

    # ----------------- 时间同步的目标/朝向（供奖励分项使用） -----------------
    def _time_sync_target(self) -> torch.Tensor:
        t = self._curr_time()
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return self._traj_gen.calc_pos(env_ids, t)                                   # [N,3]

    def _time_sync_heading(self) -> torch.Tensor:
        # 速度方向作为期望朝向（差分）
        t  = self._curr_time()
        dt = 0.1
        env_ids  = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        pos_curr = self._traj_gen.calc_pos(env_ids, t)
        pos_next = self._traj_gen.calc_pos(env_ids, t + dt)
        d = pos_next[:, :2] - pos_curr[:, :2]
        return torch.atan2(d[:, 1], d[:, 0])                                         # [N]

    # ----------------- 奖励分项（由父类聚合） -----------------
    # 对应 cfg.rewards.scales.traj_tracking
    def _reward_traj_tracking(self):
        # 与 HumanoidTraj 的 compute_traj_reward 对齐：exp(-2 * ||Δxy||^2)
        tar = self._time_sync_target()                                               # [N,3]
        diff = tar[:, :2] - self.base_pos[:, :2]
        pos_err = torch.sum(diff * diff, dim=-1)
        return torch.exp(-2.0 * pos_err)

    # 对应 cfg.rewards.scales.traj_orientation（可按需在 cfg 打开/关闭）
    def _reward_traj_orientation(self):
        target_heading = self._time_sync_heading()
        base_heading   = _quat_to_yaw_xyzw(self.base_quat)
        err = torch.abs(wrap_to_pi(base_heading - target_heading))
        return torch.exp(-2.0 * err)

    # 对应 cfg.rewards.scales.smooth_motion（可按需在 cfg 打开/关闭）
    def _reward_smooth_motion(self):
        if hasattr(self, "last_actions"):
            delta = torch.norm(self.actions - self.last_actions, dim=-1)
        else:
            delta = torch.norm(self.actions, dim=-1)
        return torch.exp(-5.0 * delta)

    # ----------------- Reset / 可视化 -----------------
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        self._regenerate_trajectories(env_ids)

        if self._is_eval:
            self._success_buf[env_ids]   = 0
            self._precision_buf[env_ids] = 0.0

    def post_physics_step(self):
        super().post_physics_step()
        if getattr(self.cfg.traj, "enable_markers", False) and hasattr(self, "update_markers"):
            k = min(getattr(self, "num_markers", 0), self.cfg.traj.num_samples)
            if k > 0:
                markers = self._fetch_marker_samples(k)
                markers[..., 2] = self.cfg.rewards.base_height_target
                self.update_markers(markers)

        if self._is_eval:
            self._compute_evaluation_metrics()

    # 追加偏离失败（对齐参考逻辑）
    def _compute_reset(self):
        super()._compute_reset()
        time   = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar    = self._traj_gen.calc_pos(env_ids, time)

        delta = tar[:, :2] - self.base_pos[:, :2]
        far   = torch.sum(delta * delta, dim=-1) > (self._fail_dist * self._fail_dist)

        if self.reset_buf.dtype == torch.bool:
            self.reset_buf |= far
        else:
            self.reset_buf = torch.where(far, torch.ones_like(self.reset_buf), self.reset_buf)

    # 评估（可选）
    def _compute_evaluation_metrics(self):
        root_pos = self.base_pos
        env_ids  = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        coeff = 0.98
        final_t = torch.full((self.num_envs,), self.max_episode_length * coeff * self.dt,
                             device=self.device, dtype=torch.float)
        traj_final = self._traj_gen.calc_pos(env_ids, final_t)
        traj_curr  = self._traj_gen.calc_pos(env_ids, self._curr_time())

        # 成功率：到最终目标≤阈值且时间已到 0.98*episode
        dist_final = torch.norm(traj_final[:, :2] - root_pos[:, :2], dim=-1)
        success    = torch.logical_and(dist_final <= self._success_threshold,
                                       self.progress_buf >= self.max_episode_length * coeff)
        self._success_buf[success] += 1

        # 精度累加：当前目标误差
        err_curr = torch.norm(traj_curr[:, :2] - root_pos[:, :2], dim=-1)
        self._precision_buf += err_curr

    # 任务观测尺寸（2*S）
    def get_task_obs_size(self):
        if getattr(self.cfg.env, "enableTaskObs", False):
            S = min(self.cfg.traj.num_samples, getattr(self.cfg.traj, "num_obs_samples", self.cfg.traj.num_samples))
            return 2 * S
        return 0