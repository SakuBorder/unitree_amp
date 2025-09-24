import torch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from phc.phc.utils import torch_utils

from legged_gym.envs.g1.g1_env import G1Robot


class G1ClimbRobot(G1Robot):
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

    def _compute_task_obs(self):
        rel_object_pos = self.climb_object_pos - self.base_pos
        rel_target_pos = self.climb_target_pos - self.base_pos

        heading_rot = torch_utils.calc_heading_quat_inv(self.base_quat)
        local_object_pos = quat_rotate(heading_rot, rel_object_pos)
        local_target_pos = quat_rotate(heading_rot, rel_target_pos)

        bbox_points = self._compute_bbox_points()
        local_bbox_points = self._transform_bbox_to_local(bbox_points, heading_rot)

        task_obs = torch.cat([
            local_target_pos,
            local_object_pos,
            self.climb_object_rot,
            local_bbox_points.reshape(self.num_envs, -1)[:, :24],
        ], dim=-1)
        return task_obs

    def _compute_bbox_points(self):
        half_size = self.climb_object_bbox / 2.0
        corners = torch.tensor(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        corners = corners.unsqueeze(0) * half_size.unsqueeze(1)

        rot = self.climb_object_rot.unsqueeze(1).expand(-1, 8, -1)
        world_corners = quat_rotate(rot.reshape(-1, 4), corners.reshape(-1, 3))
        world_corners = world_corners.reshape(self.num_envs, 8, 3)
        world_corners += self.climb_object_pos.unsqueeze(1)
        return world_corners

    def _transform_bbox_to_local(self, bbox_points, heading_rot):
        rel_points = bbox_points - self.base_pos.unsqueeze(1)
        heading_exp = heading_rot.unsqueeze(1).expand(-1, bbox_points.shape[1], -1)
        local_points = quat_rotate(
            heading_exp.reshape(-1, 4),
            rel_points.reshape(-1, 3),
        )
        return local_points.reshape(self.num_envs, -1, 3)

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

        self.climb_object_states[env_ids, 0] = torch_rand_float(1.5, 3.0, (env_ids.numel(),), device=self.device)
        self.climb_object_states[env_ids, 1] = torch_rand_float(-0.5, 0.5, (env_ids.numel(),), device=self.device)
        self.climb_object_states[env_ids, 2] = 0.0
        self.climb_object_states[env_ids, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
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
