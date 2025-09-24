import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_mul, quat_rotate, quat_from_euler_xyz

from phc.phc.utils import torch_utils

from legged_gym.envs.g1.g1_amp_env import G1AMPRobot


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


def _compute_sit_observations(
    root_states: torch.Tensor,
    tar_pos: torch.Tensor,
    object_states: torch.Tensor,
    object_bps: torch.Tensor,
    object_facings: torch.Tensor,
) -> torch.Tensor:
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_pos = quat_rotate(heading_rot, tar_pos - root_pos)

    obj_pos = object_states[:, 0:3]
    obj_rot = object_states[:, 3:7]

    local_object_pos = quat_rotate(heading_rot, obj_pos - root_pos)
    local_object_rot = quat_mul(heading_rot, obj_rot)
    local_object_rot_obs = torch_utils.quat_to_tan_norm(local_object_rot)

    obj_pos_exp = obj_pos.unsqueeze(1).expand_as(object_bps)
    obj_rot_exp = obj_rot.unsqueeze(1).expand_as(obj_pos_exp)
    root_pos_exp = root_pos.unsqueeze(1).expand_as(object_bps)
    heading_rot_exp = heading_rot.unsqueeze(1).expand_as(object_bps)

    obj_bps_world = quat_rotate(
        obj_rot_exp.reshape(-1, 4),
        object_bps.reshape(-1, 3),
    ).reshape_as(object_bps) + obj_pos_exp

    obj_bps_local = quat_rotate(
        heading_rot_exp.reshape(-1, 4),
        (obj_bps_world - root_pos_exp).reshape(-1, 3),
    ).reshape(root_pos.shape[0], -1)

    face_vec_world = quat_rotate(obj_rot, object_facings)
    face_vec_local = quat_rotate(heading_rot, face_vec_world)[..., 0:2]

    return torch.cat(
        [
            local_tar_pos,
            obj_bps_local,
            face_vec_local,
            local_object_pos,
            local_object_rot_obs,
        ],
        dim=-1,
    )


class G1SitRobot(G1AMPRobot):
    """Unitree G1 坐下交互任务。"""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.extra_actors_per_env = 1
        self._env_seat_configs = []
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self._seat_actor_rows = self.get_extra_actor_row_indices(0)

    # ------------------------------------------------------------------
    # 资产与环境构建
    # ------------------------------------------------------------------
    def _create_envs(self):
        self._load_seat_assets()
        super()._create_envs()

    def _load_seat_assets(self):
        self.seat_assets = []
        self.seat_configs = []

        asset_opts = gymapi.AssetOptions()
        asset_opts.fix_base_link = True
        asset_opts.density = 1000.0
        asset_opts.angular_damping = 0.01
        asset_opts.linear_damping = 0.01

        for cat in self.cfg.env.objCategories:
            if cat == "chair":
                asset, cfg = self._create_chair_asset(asset_opts)
            elif cat == "stool":
                asset, cfg = self._create_stool_asset(asset_opts)
            elif cat == "bench":
                asset, cfg = self._create_bench_asset(asset_opts)
            else:
                asset, cfg = self._create_default_seat_asset(asset_opts)

            self.seat_assets.append(asset)
            self.seat_configs.append(cfg)

    def _create_chair_asset(self, options):
        seat_width, seat_depth, seat_height = 0.5, 0.5, 0.45
        asset = self.gym.create_box(self.sim, seat_width, seat_depth, seat_height, options)
        cfg = {
            "seat_height": seat_height,
            "bbox": [seat_width, seat_depth, seat_height],
            "facing": [1.0, 0.0, 0.0],
            "sit_offset": [0.0, 0.0, seat_height + 0.1],
        }
        return asset, cfg

    def _create_stool_asset(self, options):
        radius, height = 0.3, 0.4
        asset = self.gym.create_box(self.sim, radius * 2, radius * 2, height, options)
        cfg = {
            "seat_height": height,
            "bbox": [radius * 2, radius * 2, height],
            "facing": [1.0, 0.0, 0.0],
            "sit_offset": [0.0, 0.0, height + 0.1],
        }
        return asset, cfg

    def _create_bench_asset(self, options):
        width, depth, height = 1.2, 0.4, 0.42
        asset = self.gym.create_box(self.sim, width, depth, height, options)
        cfg = {
            "seat_height": height,
            "bbox": [width, depth, height],
            "facing": [1.0, 0.0, 0.0],
            "sit_offset": [0.0, 0.0, height + 0.1],
        }
        return asset, cfg

    def _create_default_seat_asset(self, options):
        return self._create_chair_asset(options)

    def _build_env(self, env_id, env_ptr, robot_asset):
        super()._build_env(env_id, env_ptr, robot_asset)

        seat_asset_id = env_id % len(self.seat_assets)
        seat_asset = self.seat_assets[seat_asset_id]
        seat_cfg = self.seat_configs[seat_asset_id]

        seat_pose = gymapi.Transform()
        seat_pose.p = gymapi.Vec3(1.0, 0.0, seat_cfg["bbox"][2] / 2.0)
        seat_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.gym.create_actor(
            env_ptr,
            seat_asset,
            seat_pose,
            f"seat_{env_id}",
            env_id,
            0,
            0,
        )

        self._env_seat_configs.append(seat_cfg)

    # ------------------------------------------------------------------
    # 缓冲区与观测
    # ------------------------------------------------------------------
    def _init_buffers(self):
        super()._init_buffers()

        self.seat_states = self.get_extra_actor_state_view(0)
        self.seat_pos = self.seat_states[:, 0:3]
        self.seat_rot = self.seat_states[:, 3:7]

        configs = [
            cfg if cfg is not None else self.seat_configs[0]
            for cfg in self._env_seat_configs
        ]
        self.seat_heights = torch.tensor([cfg["seat_height"] for cfg in configs], device=self.device, dtype=torch.float32)
        self.seat_bbox = torch.tensor([cfg["bbox"] for cfg in configs], device=self.device, dtype=torch.float32)
        self.seat_facing = torch.tensor([cfg["facing"] for cfg in configs], device=self.device, dtype=torch.float32)
        self.seat_sit_offset = torch.tensor([cfg["sit_offset"] for cfg in configs], device=self.device, dtype=torch.float32)
        self._bbox_corners = _BBOX_CORNERS.to(self.device)
        self.seat_bps = 0.5 * self.seat_bbox.unsqueeze(1) * self._bbox_corners.unsqueeze(0)

        self.sit_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.sit_target_orient = torch.zeros(self.num_envs, 4, device=self.device)
        self.sit_state_buf = torch.zeros(self.num_envs, device=self.device)
        self.sit_stability_buf = torch.zeros(self.num_envs, device=self.device)

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_seats(env_ids)

    def compute_observations(self):
        super().compute_observations()
        if self.cfg.env.enableTaskObs:
            task_obs = self._compute_task_obs()
            self.obs_buf = torch.cat([self.obs_buf, task_obs], dim=-1)
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, task_obs], dim=-1)
        self._refresh_observation_extras()

    def _compute_task_obs(self):
        base_obs = _compute_sit_observations(
            self.root_states,
            self.sit_target_pos,
            self.seat_states,
            self.seat_bps,
            self.seat_facing,
        )
        return torch.cat([base_obs, self.sit_stability_buf.unsqueeze(-1)], dim=-1)

    # ------------------------------------------------------------------
    # 状态更新/重置
    # ------------------------------------------------------------------
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        self._reset_seats(env_ids)

    def _reset_seats(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        seat_height = self.seat_heights[env_ids]

        # 随机化座椅的位置和朝向
        self.seat_states[env_ids, 0] = torch.rand(env_ids.numel(), device=self.device) * 0.7 + 0.8
        self.seat_states[env_ids, 1] = torch.rand(env_ids.numel(), device=self.device) * 0.8 - 0.4
        self.seat_states[env_ids, 2] = seat_height / 2.0

        yaw = torch.rand(env_ids.numel(), device=self.device) - 0.5
        quat = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        self.seat_states[env_ids, 3:7] = quat
        self.seat_states[env_ids, 7:13] = 0.0

        # 目标位置/朝向
        seat_rot = self.seat_rot[env_ids]
        sit_offset = quat_rotate(seat_rot, self.seat_sit_offset[env_ids])
        self.sit_target_pos[env_ids] = self.seat_pos[env_ids] + sit_offset
        self.sit_target_orient[env_ids] = seat_rot

        self.sit_state_buf[env_ids] = 0.0
        self.sit_stability_buf[env_ids] = 0.0

        self.set_extra_actor_states(0, env_ids)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_sit_targets()
        self._check_sitting_state()
        self._update_stability()

    def _update_sit_targets(self):
        sit_offset_world = quat_rotate(self.seat_rot, self.seat_sit_offset)
        self.sit_target_pos = self.seat_pos + sit_offset_world
        self.sit_target_orient = self.seat_rot

    def _check_sitting_state(self):
        pos_err = torch.norm(self.base_pos - self.sit_target_pos, dim=-1)
        height_err = torch.abs(self.base_pos[:, 2] - self.sit_target_pos[:, 2])
        vel_norm = torch.norm(self.base_lin_vel[:, :2], dim=-1)

        sitting = (pos_err < 0.35) & (height_err < 0.2) & (vel_norm < 0.5)
        self.sit_state_buf = sitting.float()

    def _update_stability(self):
        vel_norm = torch.norm(self.base_lin_vel, dim=-1)
        ang_vel_norm = torch.norm(self.base_ang_vel, dim=-1)
        stability_score = torch.exp(-vel_norm) * torch.exp(-0.5 * ang_vel_norm)
        # 指数平均，增强平滑性
        self.sit_stability_buf = 0.9 * self.sit_stability_buf + 0.1 * stability_score

    # ------------------------------------------------------------------
    # 奖励函数
    # ------------------------------------------------------------------
    def _reward_sit_position(self):
        pos_err = torch.norm(self.base_pos - self.sit_target_pos, dim=-1)
        return torch.exp(-3.0 * pos_err)

    def _reward_sit_orientation(self):
        heading = torch_utils.calc_heading_quat(self.base_quat)
        seat_heading = torch_utils.calc_heading_quat(self.sit_target_orient)
        diff = torch.abs(torch_utils.wrap_to_pi(heading - seat_heading))
        return torch.exp(-2.0 * diff)

    def _reward_stable_sitting(self):
        return self.sit_stability_buf * self.sit_state_buf
