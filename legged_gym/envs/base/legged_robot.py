from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False

        # ===== Marker 开关与数量（KISS）=====
        self.enable_markers: bool = bool(getattr(self.cfg, "traj", None) and getattr(self.cfg.traj, "enable_markers", False))
        self.num_markers: int = int(getattr(self.cfg.traj, "num_samples", 0) if self.enable_markers else 0)

        # ===== 额外 actor 管理 =====
        # 子类（如携带、攀爬、坐下任务）可在 super().__init__ 之前设置 extra_actors_per_env。
        # 这里读取并规范化该属性，确保 LeggedRobot 统一维护 actors_per_env。
        self.extra_actors_per_env: int = int(getattr(self, "extra_actors_per_env", 0))

        # 必须在 create_sim 前确定 actors_per_env
        base_actors = 1 + (self.num_markers if self.enable_markers else 0)
        self.actors_per_env = base_actors + self.extra_actors_per_env
        print(
            "[LeggedRobot] actors_per_env="
            f"{self.actors_per_env} (markers={self.num_markers}, extra={self.extra_actors_per_env})"
        )

        # 这些在 _create_envs/_init_buffers 中赋值
        self.marker_asset = None
        self.marker_handles = []          # List[List[int]] per env
        self.marker_actor_ids = None      # (num_envs * num_markers,) SIM 行号
        self._root_state_gymtensor = None # flat root-state (all actors)
        self._root_state_ptr = None
        self.all_root_states = None       # view [E, A, 13]
        self.marker_states = None         # view [E, M, 13]
        self.marker_pos = None            # view [E, M, 3]
        self.robot_actor_ids = []         # (num_envs,) SIM 行号
        self.robot_rows = None            # torch int32 (num_envs,)
        self._actor_row_base = None       # base rows for each env (int32)

        # 初始化域随机化所需的张量（参考文档3）
        self.num_actions = getattr(cfg.env, 'num_actions', 12)  # 默认值
        
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    # ======================== 主循环 ========================
    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.progress_buf += 1
        self.common_step_counter += 1

        # robot 根状态（all_root_states[:,0]）
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations()
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    # ======================== 终止与重置 ========================
    def check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        
        # 越界检查（参考文档3）
        if len(env_ids) > 0:
            bad = (env_ids < 0) | (env_ids >= self.num_envs)
            if torch.any(bad):
                print(f"[FATAL] env_ids out of range: min={int(env_ids.min())}, max={int(env_ids.max())}, num_envs={self.num_envs}")
                env_ids = env_ids[~bad]
        if len(env_ids) == 0:
            return

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # 统计
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    # ======================== 观测/奖励 ========================
    def compute_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel  * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # ======================== 仿真创建 ========================
    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()

    def set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ======================== Callbacks ========================
    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        # 域随机化功能（参考文档3的实现）
        if hasattr(self.cfg.domain_rand, 'randomize_calculated_torque') and self.cfg.domain_rand.randomize_calculated_torque:
            self.torque_multiplier[env_id,:] = torch_rand_float(
                self.cfg.domain_rand.torque_multiplier_range[0], 
                self.cfg.domain_rand.torque_multiplier_range[1], 
                (1,self.num_actions), device=self.device)

        if hasattr(self.cfg.domain_rand, 'randomize_motor_zero_offset') and self.cfg.domain_rand.randomize_motor_zero_offset:
            self.motor_zero_offsets[env_id, :] = torch_rand_float(
                self.cfg.domain_rand.motor_zero_offset_range[0], 
                self.cfg.domain_rand.motor_zero_offset_range[1], 
                (1,self.num_actions), device=self.device)
        
        if hasattr(self.cfg.domain_rand, 'randomize_pd_gains') and self.cfg.domain_rand.randomize_pd_gains:
            self.p_gains_multiplier[env_id, :] = torch_rand_float(
                self.cfg.domain_rand.stiffness_multiplier_range[0], 
                self.cfg.domain_rand.stiffness_multiplier_range[1], 
                (1,self.num_actions), device=self.device)
            self.d_gains_multiplier[env_id, :] = torch_rand_float(
                self.cfg.domain_rand.damping_multiplier_range[0], 
                self.cfg.domain_rand.damping_multiplier_range[1], 
                (1,self.num_actions), device=self.device)

        return props

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _post_physics_step_callback(self):
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

    def _resample_commands(self, env_ids):
        if len(env_ids)==0:
            return
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    # ======================== 控制与复位 ========================
    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """参考文档3的实现，使用全量提交避免GPU稀疏写越界"""
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.
        
        # 使用全量提交（参考文档3）
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.refresh_dof_state_tensor(self.sim)

    def _reset_root_states(self, env_ids):
        """修正版本：参考文档3的实现，增加完整的错误检查和处理"""
        if len(env_ids) == 0:
            return
            
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            # 确保device和dtype正确
            if env_ids.device != self.device:
                env_ids = env_ids.to(self.device)
            if env_ids.dtype != torch.long:
                env_ids = env_ids.to(torch.long)

        # 机器人行号（SIM 域）- 参考文档3的逻辑
        if hasattr(self, "robot_rows") and self.robot_rows is not None and len(self.robot_rows) == self.num_envs:
            actor_rows = self.robot_rows.index_select(0, env_ids).to(dtype=torch.int32)
        else:
            actor_rows = (env_ids.to(dtype=torch.int32) * self.actors_per_env)

        # 越界保护（参考文档3）
        num_total_actors = self._root_state_gymtensor.shape[0]
        if actor_rows.numel() > 0:
            min_row = int(actor_rows.min().item())
            max_row = int(actor_rows.max().item())
            if min_row < 0 or max_row >= num_total_actors:
                print(f"[FATAL] robot_rows out of range: min={min_row}, max={max_row}, total={num_total_actors}")
                return

        # 生成初始状态
        B = env_ids.shape[0]
        base_states = self.base_init_state.unsqueeze(0).expand(B, -1).clone()
        base_states[:, :3] += self.env_origins.index_select(0, env_ids)
        if self.custom_origins:
            base_states[:, :2] += torch_rand_float(-1., 1., (B, 2), device=self.device)
        base_states[:, 7:13] = torch_rand_float(-0.5, 0.5, (B, 6), device=self.device)

        # 写入扁平 root-state（确保device匹配）
        root_flat = self._root_state_gymtensor
        if actor_rows.device != root_flat.device:
            actor_rows = actor_rows.to(root_flat.device)
        root_flat.index_copy_(0, actor_rows.long(), base_states)

        # 提交
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            self._root_state_ptr,
            gymtorch.unwrap_tensor(actor_rows),
            actor_rows.shape[0]
        )
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def _push_robots(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
        
        # 使用正确的robot_rows（参考文档3）
        if hasattr(self, "robot_rows") and self.robot_rows is not None:
            actor_rows = self.robot_rows.index_select(0, push_env_ids.to(torch.long)).to(dtype=torch.int32)
        else:
            actor_rows = (push_env_ids.to(dtype=torch.int32) * self.actors_per_env)
            
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            self._root_state_ptr,
            gymtorch.unwrap_tensor(actor_rows),
            actor_rows.shape[0]
        )
        self.gym.refresh_actor_root_state_tensor(self.sim)

    # ======================== 噪声与缓存 ========================
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """修正版本：参考文档3的实现逻辑"""
        # 获取 gym 原生张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # flat root-state & 指针
        self._root_state_gymtensor = gymtorch.wrap_tensor(actor_root_state)
        self._root_state_ptr = gymtorch.unwrap_tensor(self._root_state_gymtensor)
        self.root_state_tensor = self._root_state_gymtensor

        # 验证actors_per_env设置（参考文档3）
        if not hasattr(self, 'actors_per_env'):
            self.actors_per_env = 1
            print("Warning: actors_per_env not set, defaulting to 1")
        expected = self.num_envs * self.actors_per_env
        actual = self.root_state_tensor.shape[0]
        print(f"_init_buffers: num_envs={self.num_envs}, actors_per_env={self.actors_per_env}")
        print(f"Root tensor shape: {self.root_state_tensor.shape}, expected: {expected}")
        if actual != expected:
            raise RuntimeError(
                f"Root tensor shape mismatch! Expected {expected}, got {actual}. "
                f"Check actor creation per env (robot + markers + extra)."
            )

        # 视图: [E, A, 13]
        self.all_root_states = self.root_state_tensor.view(self.num_envs, self.actors_per_env, 13)
        self.root_states = self.all_root_states[:, 0]
        if self.enable_markers and self.num_markers > 0:
            self.marker_states = self.all_root_states[:, 1:1+self.num_markers]
            self.marker_pos = self.marker_states[..., :3]
            # 初始化为单位四元数、零速度
            marker_identity = self.root_states.new_tensor((0.0, 0.0, 0.0, 1.0))
            self.marker_states[..., 3:7] = marker_identity
            self.marker_states[..., 7:13] = 0.0

        device = self.device if hasattr(self, "device") else self.sim_device
        # 正确初始化robot_rows（参考文档3）
        if hasattr(self, "robot_actor_ids") and len(self.robot_actor_ids) == self.num_envs:
            self.robot_rows = torch.as_tensor(self.robot_actor_ids, device=device, dtype=torch.int32).contiguous()
        else:
            self.robot_rows = (torch.arange(self.num_envs, device=device, dtype=torch.int32) * self.actors_per_env)

        # 每个环境在 root tensor 中的起始行号。
        self._actor_row_base = torch.arange(
            self.num_envs, device=device, dtype=torch.int32
        ) * self.actors_per_env

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # 常用切片
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        # 缓冲
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 初始化域随机化所需的张量（参考文档3）
        self.torque_multiplier   = torch.ones(self.num_envs, self.num_actions, device=self.device)
        self.motor_zero_offsets  = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.p_gains_multiplier  = torch.ones(self.num_envs, self.num_actions, device=self.device)
        self.d_gains_multiplier  = torch.ones(self.num_envs, self.num_actions, device=self.device)

        # 默认关节与 PD
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # 可选调试信息（参考文档3）
    def _prepare_reward_function(self):
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, '_reward_' + name))
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    # ======================== 地面与环境 ========================
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _load_marker_asset(self):
        # 默认路径：LEGGED_GYM_ROOT_DIR/TokenHSI/tokenhsi/data/assets/mjcf/location_marker.urdf
        asset_root = os.path.join(LEGGED_GYM_ROOT_DIR, "TokenHSI", "tokenhsi", "data", "assets", "mjcf")
        asset_file = "location_marker.urdf"
        opts = gymapi.AssetOptions()
        opts.angular_damping = 0.01
        opts.linear_damping = 0.01
        opts.max_angular_velocity = 100.0
        opts.density = 1.0
        opts.fix_base_link = True
        opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        try:
            self.marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, opts)
        except Exception as e:
            print(f"[Marker][WARN] load asset failed: {e}. Fallback to small box.")
            # 兜底小方块
            self.marker_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, opts)

    def _create_envs(self):
        """ 修正版本：参考文档3的完整实现，确保actor创建顺序正确 """
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

        # 名称
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        
        # 确保num_actions正确设置
        if not hasattr(self, 'num_actions') or self.num_actions != self.num_dof:
            self.num_actions = self.num_dof

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.envs = []
        self.actor_handles = []
        self.robot_actor_ids = []
        self.marker_handles = [[] for _ in range(self.num_envs)] if self.enable_markers and self.num_markers > 0 else []

        if self.enable_markers and self.num_markers > 0:
            self._load_marker_asset()

        grid = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, grid)
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            robot_sim_id = self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM)
            self.robot_actor_ids.append(int(robot_sim_id))

            # ---- 创建 markers: 依次追加在 robot 后面（SIM 排列）----
            if self.enable_markers and self.num_markers > 0:
                default_pose = gymapi.Transform()
                # 初始高度对齐机器人根
                default_pose.p = gymapi.Vec3(pos[0].item(), pos[1].item(), pos[2].item())
                for _ in range(self.num_markers):
                    mhandle = self.gym.create_actor(env_handle, self.marker_asset, default_pose, "marker", i + self.num_envs + 10, 1, 0)
                    self.marker_handles[i].append(mhandle)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # 计算 marker 的 SIM 行号（假设创建顺序：robot -> markers）
        if self.enable_markers and self.num_markers > 0:
            base_rows = torch.arange(self.num_envs, device=self.device, dtype=torch.int32) * self.actors_per_env
            # shape: [E, M]，每 env 的 markers 行：base+1 ... base+M
            row_grid = base_rows.unsqueeze(1) + 1 + torch.arange(self.num_markers, device=self.device, dtype=torch.int32).unsqueeze(0)
            self.marker_actor_ids = row_grid.reshape(-1).contiguous()

        # 刚体索引（ENV 域）
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, n in enumerate(feet_names):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_index(self.envs[0], self.actor_handles[0], n, gymapi.DOMAIN_ENV)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, n in enumerate(penalized_contact_names):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_index(self.envs[0], self.actor_handles[0], n, gymapi.DOMAIN_ENV)

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, n in enumerate(termination_contact_names):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_index(self.envs[0], self.actor_handles[0], n, gymapi.DOMAIN_ENV)

    def _get_env_origins(self):
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    # ======================== 简要奖励函数 ========================
    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    # ======================== 额外 actor 工具 ========================
    def _ensure_actor_row_base(self):
        if self._actor_row_base is None or self._actor_row_base.shape[0] != self.num_envs:
            device = self.device if hasattr(self, "device") else "cpu"
            self._actor_row_base = (
                torch.arange(self.num_envs, device=device, dtype=torch.int32)
                * self.actors_per_env
            )

    def get_actor_row_indices(self, actor_offset: int) -> torch.Tensor:
        self._ensure_actor_row_base()
        offset = int(actor_offset)
        if offset < 0 or offset >= self.actors_per_env:
            raise ValueError(f"actor_offset {actor_offset} out of range [0, {self.actors_per_env})")
        return self._actor_row_base + offset

    def get_extra_actor_offset(self, extra_index: int) -> int:
        if self.extra_actors_per_env <= 0:
            raise RuntimeError("No extra actors registered in this environment")
        if extra_index < 0 or extra_index >= self.extra_actors_per_env:
            raise IndexError(
                f"extra_index {extra_index} invalid for {self.extra_actors_per_env} extra actors"
            )
        return self.actors_per_env - self.extra_actors_per_env + extra_index

    def get_extra_actor_row_indices(self, extra_index: int) -> torch.Tensor:
        offset = self.get_extra_actor_offset(extra_index)
        return self.get_actor_row_indices(offset)

    def get_extra_actor_state_view(self, extra_index: int) -> torch.Tensor:
        offset = self.get_extra_actor_offset(extra_index)
        return self.all_root_states[:, offset]

    def set_extra_actor_states(self, extra_index: int, env_ids, new_states: torch.Tensor = None):
        if self.extra_actors_per_env <= 0:
            raise RuntimeError("set_extra_actor_states called but no extra actors are configured")

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        if env_ids.numel() == 0:
            return

        offset = self.get_extra_actor_offset(extra_index)
        if new_states is not None:
            new_states = new_states.to(device=self.device)
            self.all_root_states[env_ids, offset] = new_states

        rows = self.get_actor_row_indices(offset).index_select(0, env_ids).contiguous()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            self._root_state_ptr,
            gymtorch.unwrap_tensor(rows),
            rows.shape[0],
        )
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def update_markers(self, world_xyz: torch.Tensor):
        """
        world_xyz: [num_envs, num_markers, 3] 世界系坐标。
        将 Z 对齐到当前 robot 根高，然后局部写回到对应 marker actors。
        """
        if not (self.enable_markers and self.num_markers > 0):
            return
        if world_xyz is None or world_xyz.shape[:2] != (self.num_envs, self.num_markers):
            raise ValueError(f"world_xyz shape must be [{self.num_envs}, {self.num_markers}, 3]")

        # 写入视图
        self.marker_pos[:] = world_xyz.to(device=self.device)
        # Z 对齐 base 高度（视觉一致性）
        base_h = self.root_states[:, 2:3]
        self.marker_pos[..., 2] = base_h

        # 提交局部更新
        if self.marker_actor_ids is None or self.marker_actor_ids.numel() == 0:
            return
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state_gymtensor),
            gymtorch.unwrap_tensor(self.marker_actor_ids),
            int(self.marker_actor_ids.numel()),
        )