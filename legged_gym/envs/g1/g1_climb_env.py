import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate

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


def _compute_climb_observations(
    root_states: torch.Tensor,
    climb_object_states: torch.Tensor,
    climb_object_bps: torch.Tensor,
    climb_tar_pos: torch.Tensor,
) -> torch.Tensor:
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    heading_rot_exp = heading_rot.unsqueeze(1).expand_as(climb_object_bps)
    root_pos_exp = root_pos.unsqueeze(1).expand_as(climb_object_bps)

    obj_pos = climb_object_states[:, 0:3]
    obj_rot = climb_object_states[:, 3:7]
    obj_pos_exp = obj_pos.unsqueeze(1).expand_as(climb_object_bps)
    obj_rot_exp = obj_rot.unsqueeze(1).expand_as(heading_rot_exp)

    obj_bps_world = quat_rotate(
        obj_rot_exp.reshape(-1, 4),
        climb_object_bps.reshape(-1, 3),
    ).reshape_as(climb_object_bps) + obj_pos_exp

    obj_bps_local = quat_rotate(
        heading_rot_exp.reshape(-1, 4),
        (obj_bps_world - root_pos_exp).reshape(-1, 3),
    ).reshape(root_pos.shape[0], -1)

    local_climb_tar = quat_rotate(heading_rot, climb_tar_pos - root_pos)

    return torch.cat([local_climb_tar, obj_bps_local], dim=-1)


class G1ClimbRobot(G1AMPRobot):
    """Unitree G1 攀爬交互任务。"""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 额外的攀爬物体 actor
        self.extra_actors_per_env = 1
        self._env_climb_bbox = []
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self._climb_actor_rows = self.get_extra_actor_row_indices(0)

    # ------------------------------------------------------------------
    # 资产与环境创建
    # ------------------------------------------------------------------
    def _create_envs(self):
        self._create_climb_assets()
        super()._create_envs()

    def _create_climb_assets(self):
        self.climb_assets = []
        self.climb_asset_bbox = []

        asset_opts = gymapi.AssetOptions()
        asset_opts.fix_base_link = True
        asset_opts.density = 1000.0
        asset_opts.angular_damping = 0.01
        asset_opts.linear_damping = 0.01

        for cat in self.cfg.env.objCategories:
            if cat == "stairs":
                width, depth, height, num_steps = 1.5, 0.3, 0.15, 4
                asset = self.gym.create_box(
                    self.sim,
                    width,
                    depth * num_steps,
                    height * num_steps,
                    asset_opts,
                )
                bbox = (width, depth * num_steps, height * num_steps)
            elif cat == "ramp":
                ramp_width, ramp_length, ramp_height = 1.5, 2.0, 0.5
                asset = self.gym.create_box(
                    self.sim,
                    ramp_width,
                    ramp_length,
                    ramp_height,
                    asset_opts,
                )
                bbox = (ramp_width, ramp_length, ramp_height)
            elif cat == "obstacle":
                asset = self.gym.create_box(self.sim, 1.0, 1.0, 0.4, asset_opts)
                bbox = (1.0, 1.0, 0.4)
            else:
                asset = self.gym.create_box(self.sim, 1.0, 0.5, 0.5, asset_opts)
                bbox = (1.0, 0.5, 0.5)

            self.climb_assets.append(asset)
            self.climb_asset_bbox.append(torch.tensor(bbox, dtype=torch.float32))

    def _build_env(self, env_id, env_ptr, robot_asset):
        super()._build_env(env_id, env_ptr, robot_asset)

        asset_id = env_id % len(self.climb_assets)
        climb_asset = self.climb_assets[asset_id]
        bbox = self.climb_asset_bbox[asset_id]

        climb_pose = gymapi.Transform()
        climb_pose.p = gymapi.Vec3(2.0, 0.0, 0.0)
        climb_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.gym.create_actor(
            env_ptr,
            climb_asset,
            climb_pose,
            f"climb_object_{env_id}",
            env_id,
            0,
            0,
        )

        self._env_climb_bbox.append(bbox)

    # ------------------------------------------------------------------
    # 缓冲区与观测
    # ------------------------------------------------------------------
    def _init_buffers(self):
        super()._init_buffers()

        self.climb_object_states = self.get_extra_actor_state_view(0)
        self.climb_object_pos = self.climb_object_states[:, 0:3]
        self.climb_object_rot = self.climb_object_states[:, 3:7]

        self.climb_object_bbox = torch.stack(
            [bbox.to(self.device) for bbox in self._env_climb_bbox], dim=0
        )
        self._bbox_corners = _BBOX_CORNERS.to(self.device)
        self.climb_object_bps = 0.5 * self.climb_object_bbox.unsqueeze(1) * self._bbox_corners.unsqueeze(0)

        self.climb_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.climb_progress = torch.zeros(self.num_envs, device=self.device)
        self.prev_climb_height = torch.zeros(self.num_envs, device=self.device)

        if self.cfg.env.enableIET:
            self.IET_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.IET_triggered = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_climb_objects(env_ids)

    def compute_observations(self):
        super().compute_observations()
        if self.cfg.env.enableTaskObs:
            task_obs = self._compute_task_obs()
            self.obs_buf = torch.cat([self.obs_buf, task_obs], dim=-1)
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, task_obs], dim=-1)
        self._refresh_observation_extras()

    def _compute_task_obs(self):
        return _compute_climb_observations(
            self.root_states,
            self.climb_object_states,
            self.climb_object_bps,
            self.climb_target_pos,
        )

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
        self._reset_climb_objects(env_ids)

    def _reset_climb_objects(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        self.climb_object_states[env_ids, 0] = torch.rand(env_ids.numel(), device=self.device) * 1.5 + 1.5
        self.climb_object_states[env_ids, 1] = torch.rand(env_ids.numel(), device=self.device) - 0.5
        self.climb_object_states[env_ids, 2] = 0.0
        self.climb_object_states[env_ids, 3:7] = self.climb_object_states.new_tensor((0.0, 0.0, 0.0, 1.0))
        self.climb_object_states[env_ids, 7:13] = 0.0

        bbox = self.climb_object_bbox[env_ids]
        self.climb_target_pos[env_ids, 0:2] = self.climb_object_states[env_ids, 0:2]
        self.climb_target_pos[env_ids, 2] = bbox[:, 2] + self.cfg.rewards.base_height_target

        self.climb_progress[env_ids] = 0.0
        self.prev_climb_height[env_ids] = self.base_pos[env_ids, 2]

        if self.cfg.env.enableIET:
            self.IET_step_buf[env_ids] = 0
            self.IET_triggered[env_ids] = False

        self.set_extra_actor_states(0, env_ids)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

        denom = torch.clamp(self.climb_object_bbox[:, 2], min=0.2)
        self.climb_progress = (self.base_pos[:, 2] - self.cfg.rewards.base_height_target) / denom
        self.climb_progress = torch.clamp(self.climb_progress, 0.0, 1.0)

        if self.cfg.env.enableIET:
            self._check_iet_condition()

    def _check_iet_condition(self):
        dist_to_target = torch.norm(self.base_pos - self.climb_target_pos, dim=-1)
        success_mask = dist_to_target < self.cfg.env.successThreshold
        self.IET_step_buf[success_mask] += 1
        self.IET_step_buf[~success_mask] = 0
        self.IET_triggered = self.IET_step_buf >= self.cfg.env.maxIETSteps

    def check_termination(self):
        super().check_termination()
        if self.cfg.env.enableIET:
            self.reset_buf |= self.IET_triggered
        fell_off = self.base_pos[:, 2] < (self.cfg.rewards.base_height_target - 0.3)
        self.reset_buf |= fell_off

    # ------------------------------------------------------------------
    # 奖励函数
    # ------------------------------------------------------------------
    def _reward_climb_progress(self):
        height_progress = self.base_pos[:, 2] - self.prev_climb_height
        self.prev_climb_height[:] = self.base_pos[:, 2]
        return torch.clamp(height_progress * 10.0, -1.0, 1.0)

    def _reward_feet_clearance(self):
        feet_height = self.feet_pos[:, :, 2] - self.climb_object_pos[:, 2].unsqueeze(1)
        swing_mask = self.leg_phase > 0.5
        clearance = feet_height * swing_mask
        return torch.sum(clearance, dim=1) * 0.1

    def _reward_climb_stability(self):
        gravity_penalty = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        ang_vel_penalty = torch.sum(torch.square(self.base_ang_vel), dim=1)
        return -gravity_penalty - 0.1 * ang_vel_penalty

    def _reward_approach_object(self):
        dist = torch.norm(self.climb_object_pos[:, :2] - self.base_pos[:, :2], dim=-1)
        reward = torch.exp(-dist)
        return reward * (self.climb_progress < 0.1)

    def _reward_feet_contact_on_object(self):
        half_size = self.climb_object_bbox / 2.0
        feet_above = self.feet_pos[:, :, 2] > self.climb_object_pos[:, 2].unsqueeze(1)
        feet_in_x = torch.abs(
            self.feet_pos[:, :, 0] - self.climb_object_pos[:, 0].unsqueeze(1)
        ) < half_size[:, 0].unsqueeze(1)
        feet_in_y = torch.abs(
            self.feet_pos[:, :, 1] - self.climb_object_pos[:, 1].unsqueeze(1)
        ) < half_size[:, 1].unsqueeze(1)
        feet_on = feet_above & feet_in_x & feet_in_y
        feet_contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        valid_contact = feet_on & feet_contact
        return torch.sum(valid_contact.float(), dim=1) * 0.5
