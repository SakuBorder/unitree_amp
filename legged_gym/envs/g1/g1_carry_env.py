import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_mul, quat_rotate

from phc.phc.utils import torch_utils

from legged_gym.envs.g1.g1_env import G1Robot


_BBOX_CORNERS = torch.tensor(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=torch.float32,
)


def _compute_carry_observations(
    root_states: torch.Tensor,
    box_states: torch.Tensor,
    box_bps: torch.Tensor,
    tar_pos: torch.Tensor,
    enable_bbox_obs: bool,
) -> torch.Tensor:
    """Builds carry observations following TokenHSI's formulation."""

    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    box_pos = box_states[:, 0:3]
    box_rot = box_states[:, 3:7]
    box_vel = box_states[:, 7:10]
    box_ang_vel = box_states[:, 10:13]

    local_box_pos = quat_rotate(heading_rot, box_pos - root_pos)
    local_box_rot = quat_mul(heading_rot, box_rot)
    local_box_rot_obs = torch_utils.quat_to_tan_norm(local_box_rot)
    local_box_vel = quat_rotate(heading_rot, box_vel)
    local_box_ang_vel = quat_rotate(heading_rot, box_ang_vel)

    heading_rot_exp = heading_rot.unsqueeze(1).expand_as(box_bps)
    root_pos_exp = root_pos.unsqueeze(1).expand_as(box_bps)
    box_pos_exp = box_pos.unsqueeze(1).expand_as(box_bps)
    box_rot_exp = box_rot.unsqueeze(1).expand_as(heading_rot_exp)

    box_bps_world = quat_rotate(
        box_rot_exp.reshape(-1, 4),
        box_bps.reshape(-1, 3),
    ).reshape_as(box_bps) + box_pos_exp

    box_bps_local = quat_rotate(
        heading_rot_exp.reshape(-1, 4),
        (box_bps_world - root_pos_exp).reshape(-1, 3),
    ).reshape(root_pos.shape[0], -1)

    local_tar_pos = quat_rotate(heading_rot, tar_pos - root_pos)

    if enable_bbox_obs:
        obs_parts = [
            local_box_vel,
            local_box_ang_vel,
            local_box_pos,
            local_box_rot_obs,
            box_bps_local,
            local_tar_pos,
        ]
    else:
        obs_parts = [
            local_box_vel,
            local_box_ang_vel,
            local_box_pos,
            local_box_rot_obs,
            local_tar_pos,
        ]
    return torch.cat(obs_parts, dim=-1)


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
        self.box_ang_vel = self.box_states[:, 10:13]

        # 目标位置/缓存
        self.tar_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_box_pos = torch.zeros_like(self.box_pos)

        # "手" 的 proxy 索引（用脚部代替）
        self.hand_indices = [5, 11]

        base_size = torch.tensor(self.cfg.box.build.baseSize, device=self.device, dtype=torch.float32)
        self.box_size = base_size.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._bbox_corners = _BBOX_CORNERS.to(self.device)
        self.box_bps = 0.5 * self.box_size.unsqueeze(1) * self._bbox_corners.unsqueeze(0)

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
        enable_bbox = getattr(self.cfg.box.obs, "enableBboxObs", False)
        return _compute_carry_observations(
            self.root_states,
            self.box_states,
            self.box_bps,
            self.tar_pos,
            enable_bbox,
        )

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
        self.box_states[env_ids, 0] = torch.rand(env_ids.numel(), device=self.device) * 1.2 - 0.6
        self.box_states[env_ids, 1] = torch.rand(env_ids.numel(), device=self.device) * 1.2 - 0.6
        self.box_states[env_ids, 2] = height / 2.0
        self.box_states[env_ids, 3:7] = self.box_states.new_tensor((0.0, 0.0, 0.0, 1.0))
        self.box_states[env_ids, 7:13] = 0.0

        # 随机目标位置
        self.tar_pos[env_ids, 0] = torch.rand(env_ids.numel(), device=self.device) * 4.0 - 2.0
        self.tar_pos[env_ids, 1] = torch.rand(env_ids.numel(), device=self.device) * 4.0 - 2.0
        self.tar_pos[env_ids, 2] = height / 2.0

        self.prev_box_pos[env_ids] = self.box_pos[env_ids]

        # 更新 bbox 点
        self.box_bps[env_ids] = 0.5 * self.box_size[env_ids].unsqueeze(1) * self._bbox_corners.unsqueeze(0)

        # 写回仿真
        self.set_extra_actor_states(0, env_ids)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.prev_box_pos[:] = self.box_pos
