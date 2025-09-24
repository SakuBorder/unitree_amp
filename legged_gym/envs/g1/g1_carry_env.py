import torch
from isaacgym import gymapi
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.g1.g1_env import G1Robot


class G1CarryRobot(G1Robot):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 每个环境额外增加 1 个搬运箱体 actor。
        self.extra_actors_per_env = 1
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # 额外 actor 在 root-state tensor 中的偏移量/行索引。
        self._box_actor_offset = self.get_extra_actor_offset(0)
        self._box_actor_rows = self.get_extra_actor_row_indices(0)

    # ------------------------------------------------------------------
    # 资产与缓存初始化
    # ------------------------------------------------------------------
    def _create_envs(self):
        super()._create_envs()
        self._create_boxes()

    def _create_boxes(self):
        """为每个环境创建可搬运的箱子 actor。"""

        asset_opts = gymapi.AssetOptions()
        asset_opts.density = (
            100.0 if not getattr(self.cfg.box.build, "randomDensity", False) else 500.0
        )

        base_size = torch.tensor(self.cfg.box.build.baseSize, dtype=torch.float32)
        box_asset = self.gym.create_box(
            self.sim,
            float(base_size[0]),
            float(base_size[1]),
            float(base_size[2]),
            asset_opts,
        )

        default_pose = gymapi.Transform()
        default_pose.p = gymapi.Vec3(0.6, 0.0, float(base_size[2] / 2.0))

        self.box_handles = []
        for env_id, env_ptr in enumerate(self.envs):
            handle = self.gym.create_actor(
                env_ptr,
                box_asset,
                default_pose,
                "carry_box",
                env_id,
                0,
                0,
            )
            self.box_handles.append(handle)

    def _init_buffers(self):
        super()._init_buffers()

        # 箱体状态（view），与 root-state tensor 共享存储。
        self.box_states = self.get_extra_actor_state_view(0)
        self.box_pos = self.box_states[:, 0:3]
        self.box_rot = self.box_states[:, 3:7]
        self.box_vel = self.box_states[:, 7:10]

        # 目标位置/缓存
        self.tar_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_box_pos = torch.zeros_like(self.box_pos)

        # "手" 的 proxy 索引（用脚部代替）
        self.hand_indices = [5, 11]

        # 初始化所有环境的箱子状态
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_boxes(env_ids)

    # ------------------------------------------------------------------
    # 观测与奖励
    # ------------------------------------------------------------------
    def compute_observations(self):
        super().compute_observations()
        if self.cfg.env.enableTaskObs:
            task_obs = self._compute_task_obs()
            self.obs_buf = torch.cat([self.obs_buf, task_obs], dim=-1)
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, task_obs], dim=-1)

    def _compute_task_obs(self):
        rel_box_pos = self.box_pos - self.base_pos
        rel_tar_pos = self.tar_pos - self.base_pos

        box_rot_obs = torch.cat([
            self.box_rot[:, -1:].contiguous(),  # w
            self.box_rot[:, :3].contiguous(),
        ], dim=-1)

        return torch.cat([
            rel_tar_pos,
            rel_box_pos,
            box_rot_obs,
            self.box_vel,
        ], dim=-1)

    def _reward_carry_walk(self):
        dist = torch.norm(self.box_pos[:, :2] - self.base_pos[:, :2], dim=-1)
        return torch.exp(-0.5 * dist)

    def _reward_carry_vel(self):
        box_to_tar = self.tar_pos[:, :2] - self.box_pos[:, :2]
        norm = torch.norm(box_to_tar, dim=-1, keepdim=True)
        box_to_tar_dir = box_to_tar / (norm + 1e-6)
        vel_reward = torch.sum(box_to_tar_dir * self.box_vel[:, :2], dim=-1)
        return torch.exp(-2.0 * (1.5 - vel_reward) ** 2)

    def _reward_handheld(self):
        hand_pos = self.feet_pos[:, :, :2]
        hand_to_box = torch.norm(hand_pos.mean(dim=1) - self.box_pos[:, :2], dim=-1)
        return torch.exp(-5.0 * hand_to_box)

    def _reward_putdown(self):
        at_target = torch.norm(self.box_pos - self.tar_pos, dim=-1) < 0.1
        on_ground = torch.abs(self.box_pos[:, 2] - self.cfg.box.build.baseSize[2] / 2.0) < 0.02
        return (at_target & on_ground).float()

    # ------------------------------------------------------------------
    # 状态更新与重置
    # ------------------------------------------------------------------
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        self._reset_boxes(env_ids)

    def _reset_boxes(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        height = float(self.cfg.box.build.baseSize[2])

        # 随机起始位置/姿态
        self.box_states[env_ids, 0:2] = torch_rand_float(-0.6, 0.6, (env_ids.numel(), 2), device=self.device)
        self.box_states[env_ids, 2] = height / 2.0
        self.box_states[env_ids, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self.box_states[env_ids, 7:13] = 0.0

        # 随机目标位置
        self.tar_pos[env_ids, 0:2] = torch_rand_float(-2.0, 2.0, (env_ids.numel(), 2), device=self.device)
        self.tar_pos[env_ids, 2] = height / 2.0

        self.prev_box_pos[env_ids] = self.box_pos[env_ids]

        # 写回仿真
        self.set_extra_actor_states(0, env_ids)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.prev_box_pos[:] = self.box_pos
