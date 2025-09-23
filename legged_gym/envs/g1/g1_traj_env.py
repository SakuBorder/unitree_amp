import os

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (
    get_euler_xyz_in_tensor,
    quat_rotate_inverse,
    torch_rand_float,
)

from TokenHSI.tokenhsi.utils import traj_generator, torch_utils
from legged_gym.envs.g1.g1_amp_env import G1AMPRobot
from legged_gym import LEGGED_GYM_ROOT_DIR


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

        # marker state placeholders (populated in _create_envs / _init_buffers)
        self._traj_marker_asset = None
        self._traj_marker_handles = None
        self._traj_marker_states = None
        self._traj_marker_pos = None
        self._traj_marker_actor_ids = None
        self._all_actor_root_states = None
        self._actors_per_env = 1
        self._robot_actor_ids = None

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
        if self._traj_marker_pos is not None:
            self._update_markers_from_samples()


    # -------------------- isaac gym overrides ----------
    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = torch.tensor(base_init_state_list, device=self.device, dtype=torch.float32)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)

        self.actor_handles = []
        self.envs = []

        if not self.headless:
            self._traj_marker_handles = [[] for _ in range(self.num_envs)]
            self._load_marker_asset()

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            if self._traj_marker_handles is not None:
                self._create_marker_actors(i, env_handle)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device)
        for idx, name in enumerate(feet_names):
            self.feet_indices[idx] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for idx, name in enumerate(penalized_contact_names):
            self.penalised_contact_indices[idx] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for idx, name in enumerate(termination_contact_names):
            self.termination_contact_indices[idx] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )


    def _init_buffers(self):
        super()._init_buffers()
        # Preserve the full actor state tensor before carving out the robot slice.
        self._all_actor_root_states = self.root_states
        self._refresh_root_state_view(initial=True)
        if self._traj_marker_handles is not None:
            self._build_marker_state_tensors()
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
    def _reset_dofs(self, env_ids):
        if len(env_ids) == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.5, 1.5, (env_ids.numel(), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0

        actor_ids = self._robot_actor_ids[env_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(actor_ids),
            actor_ids.numel(),
        )

    def _reset_root_states(self, env_ids):
        if len(env_ids) == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1.0, 1.0, (env_ids.numel(), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (env_ids.numel(), 6), device=self.device)

        actor_ids = self._robot_actor_ids[env_ids].to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._all_actor_root_states),
            gymtorch.unwrap_tensor(actor_ids),
            actor_ids.numel(),
        )

    def _push_robots(self):
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        push_interval = int(self.cfg.domain_rand.push_interval)
        push_mask = self.episode_length_buf[env_ids] % push_interval == 0
        push_env_ids = env_ids[push_mask]
        if push_env_ids.numel() == 0:
            return

        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[push_env_ids, 7:9] = torch_rand_float(
            -max_vel, max_vel, (push_env_ids.numel(), 2), device=self.device
        )

        actor_ids = self._robot_actor_ids[push_env_ids].to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._all_actor_root_states),
            gymtorch.unwrap_tensor(actor_ids),
            actor_ids.numel(),
        )

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

    def _refresh_root_state_view(self, initial=False):
        if self._all_actor_root_states is None:
            actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
            self._all_actor_root_states = gymtorch.wrap_tensor(actor_root_state)

        self.gym.refresh_actor_root_state_tensor(self.sim)

        all_actor_states = self._all_actor_root_states
        self._actors_per_env = all_actor_states.shape[0] // self.num_envs
        actor_root_state_view = all_actor_states.view(
            self.num_envs, self._actors_per_env, all_actor_states.shape[-1]
        )

        self.root_states = actor_root_state_view[:, 0, :]
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._robot_actor_ids = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.int32) * self._actors_per_env
        )

        if initial or self.last_root_vel.shape[0] != self.num_envs:
            self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

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
        asset_root = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "TokenHSI",
            "tokenhsi",
            "data",
            "assets",
            "mjcf",
        )
        asset_file = "location_marker.urdf"
        opts = gymapi.AssetOptions()
        opts.angular_damping = 0.01
        opts.linear_damping = 0.01
        opts.max_angular_velocity = 100.0
        opts.density = 1.0
        opts.fix_base_link = True
        opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._traj_marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, opts)

    def _create_marker_actors(self, env_idx, env_ptr):
        if self._traj_marker_handles is None:
            return
        handles = self._traj_marker_handles[env_idx]
        default_pose = gymapi.Transform()
        col_group = self.num_envs + 10
        col_filter = 1
        segmentation_id = 0
        for _ in range(self._num_traj_samples):
            handle = self.gym.create_actor(
                env_ptr,
                self._traj_marker_asset,
                default_pose,
                "marker",
                col_group,
                col_filter,
                segmentation_id,
            )
            self.gym.set_actor_scale(env_ptr, handle, 0.5)
            self.gym.set_rigid_body_color(
                env_ptr, handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.0, 0.0)
            )
            handles.append(handle)

    def _build_marker_actor_ids(self):
        if self._traj_marker_handles is None:
            return

        handle_tensor = torch.tensor(self._traj_marker_handles, device=self.device, dtype=torch.int32)
        if self._robot_actor_ids is None:
            self._robot_actor_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32) * self._actors_per_env
        self._traj_marker_actor_ids = (self._robot_actor_ids.unsqueeze(-1) + handle_tensor).flatten()

    def _build_marker_state_tensors(self):
        if self._traj_marker_handles is None or self._all_actor_root_states is None:
            return

        marker_start = 1
        marker_end = marker_start + self._num_traj_samples
        if marker_end > self._actors_per_env:
            raise RuntimeError("Trajectory marker count exceeds actors per environment.")

        root_states_view = self._all_actor_root_states.view(
            self.num_envs, self._actors_per_env, self._all_actor_root_states.shape[-1]
        )
        self._traj_marker_states = root_states_view[:, marker_start:marker_end, :]
        self._traj_marker_pos = self._traj_marker_states[..., :3]
        marker_identity = self.root_states.new_tensor((0.0, 0.0, 0.0, 1.0))
        self._traj_marker_states[..., 3:7] = marker_identity
        self._traj_marker_states[..., 7:13] = 0.0


    def _update_markers_from_samples(self):
        if (
            self._traj_marker_pos is None
            or self._traj_marker_actor_ids is None
            or self._traj_marker_actor_ids.numel() == 0
            or self._all_actor_root_states is None
        ):
            return

        self._traj_marker_pos[:] = self._traj_samples_world
        root_height = self.root_states[:, 2:3]
        self._traj_marker_pos[..., 2] = root_height

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._all_actor_root_states),
            gymtorch.unwrap_tensor(self._traj_marker_actor_ids),
            len(self._traj_marker_actor_ids),
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
    if num_envs == 0 or num_samples == 0:
        return traj_samples.new_zeros((0, num_samples, 2))

    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(1), (num_envs, num_samples, 4))
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (num_envs, num_samples, 3))

    local_traj_samples = torch_utils.quat_rotate(heading_rot_exp.reshape(-1, 4),
                                                 traj_samples.reshape(-1, 3) - root_pos_exp.reshape(-1, 3))

    return local_traj_samples[..., 0:2].reshape(num_envs, num_samples, 2)


@torch.jit.script
def compute_traj_reward(root_pos: torch.Tensor, tar_pos: torch.Tensor) -> torch.Tensor:
    # 与 HumanoidTraj 保持一致：固定系数 2.0，仅 XY 误差
    pos_err_scale = 2.0
    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    return torch.exp(-pos_err_scale * pos_err)
