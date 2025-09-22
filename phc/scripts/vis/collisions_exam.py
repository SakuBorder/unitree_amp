import numpy as np
import yaml
import os
import sys
import time
import math
import isaacgym
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
import joblib
import mujoco
import mujoco.viewer
import torch


class Humanoid():
    def __init__(self, modelRoot, modelPath):
        self.gym = gymapi.acquire_gym()
        self.sim_params = self.parse_sim_params()
        self.dt = self.sim_params.dt
        self.up_axis = 'z'
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.physics_engine = gymapi.SIM_PHYSX
        self.device = 'cpu'

        self.show = True  # 是否开启可视化
        self.graphics_device_id = 0
        self.device_id = 0

        self.num_envs = 2  # 环境数量
        self.num_obs = 0
        self.num_states = 0
        self.num_actions = 0

        self.humanAssetFileRoot = modelRoot
        self.humanAssetFilePath = modelPath

        self.sim = self.create_sim()

        self.create_ground_plane()
        self.create_envs()

        self.gym.prepare_sim(self.sim)

        self.enable_viewer_sync = True
        self.viewer = None

        if self.show:
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            cam_pos = gymapi.Vec3(-5.0, -5, 5.0)
            cam_target = gymapi.Vec3(0, 0, 0)
            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        self.human_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_human_dofs]
        self.human_dof_pos = self.human_dof_state[..., 0]
        self.human_dof_vel = self.human_dof_state[..., 1]

        self.human_positions = self.root_state_tensor[:, 0:3]
        self.human_orientations = self.root_state_tensor[:, 3:7]
        self.human_linvels = self.root_state_tensor[:, 7:10]
        self.human_angvels = self.root_state_tensor[:, 10:13]

        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        human_color = gymapi.Vec3(0., 0., 1)
        human_color2 = gymapi.Vec3(1., 0., 0)

        motion_data = joblib.load(
            '/home/zzg/code/Ti5Robot/data/ti5robot/v1/amass_all.pkl')
        motion_data_keys = list(motion_data.keys())
        curr_motion = motion_data[motion_data_keys[2]]
        time_step = 0
        dt = 1/30
        mj_model = mujoco.MjModel.from_xml_path('/home/zzg/code/Ti5Robot/assert/ti5robot/urdf/ti.xml')
        mj_data = mujoco.MjData(mj_model)
        mj_model.opt.timestep = dt
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            while True:
                time_step += dt
                curr_time = int(time_step / dt) % curr_motion['dof'].shape[0]
                mj_data.qpos[:3] = curr_motion['root_trans_offset'][curr_time]
                # print(curr_motion['root_trans_offset'][curr_time])
                mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]]
                mj_data.qpos[7:] = curr_motion['dof'][curr_time]

                # print(f"root pos: {curr_motion['root_trans_offset'][curr_time]}")
                # print(f"root quat: {curr_motion['root_rot'][curr_time][[3, 0, 1, 2]]}")
                print(f"dof: {curr_motion['dof'][curr_time]}")

                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()

                # print(curr_time)
                self.root_state_tensor[:, 0:3] = torch.tensor(curr_motion['root_trans_offset'][curr_time]).cuda()
                self.root_state_tensor[:, 3:7] = torch.tensor(curr_motion['root_rot'][curr_time]).cuda()
                self.human_dof_pos[:, 0:30] = torch.tensor(curr_motion['dof']
                                                           [curr_time][[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                           10, 11, 12, 13, 14, 15, 16, 17,
                                                           18, 19, 20, 21, 29, 22, 23, 24,
                                                           25, 26, 27, 28]]).cuda()
                self.human_dof_vel = torch.zeros_like(self.human_dof_pos).cuda()
                self.human_linvels = torch.zeros(2, 3).cuda()
                self.human_angvels = torch.zeros(2, 3).cuda()
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))
                print(self.human_dof_pos[:, 0:30])
                # for i in range(self.contact_forces.shape[0]):
                #     for j in range(self.contact_forces.shape[1]):
                #         if torch.norm(self.contact_forces[i, j, :], dim=-1) > 1:
                #             self.gym.set_rigid_body_color(self.envs[i], self.humans[i], j, gymapi.MESH_VISUAL,
                #                                           human_color)
                #         else:
                #             self.gym.set_rigid_body_color(self.envs[i], self.humans[i], j, gymapi.MESH_VISUAL,
                #                                           human_color2)

                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_dof_state_tensor(self.sim)
                self.gym.refresh_net_contact_force_tensor(self.sim)

                self.gym.simulate(self.sim)
                self.render()

    def create_envs(self):
        human_asset_options = gymapi.AssetOptions()
        human_asset_options.fix_base_link = False
        human_asset_options.flip_visual_attachments = False
        human_asset_options.disable_gravity = False
        human_asset_options.override_com = True
        human_asset_options.override_inertia = True
        human_asset_options.thickness = 0.001
        human_asset_options.use_physx_armature = True
        human_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        human_asset = self.gym.load_asset(self.sim, self.humanAssetFileRoot,
                                          self.humanAssetFilePath, human_asset_options)

        self.num_human_bodies = self.gym.get_asset_rigid_body_count(human_asset)
        self.num_human_shapes = self.gym.get_asset_rigid_shape_count(human_asset)
        self.num_human_dofs = self.gym.get_asset_dof_count(human_asset)
        self.human_asset_dof_names = self.gym.get_asset_dof_names(human_asset)

        self.num_human_actuators = self.gym.get_asset_actuator_count(human_asset)
        self.human_asset_actuator_names = [self.gym.get_asset_actuator_name(human_asset, i) for i in
                                           range(self.num_human_actuators)]
        self.num_human_tendons = self.gym.get_asset_tendon_count(human_asset)
        self.human_asset_rigid_body_dict = self.gym.get_asset_rigid_body_dict(human_asset)
        self.sorted_human_asset_rigid_body_dict = sorted(self.human_asset_rigid_body_dict.items(), key=lambda x: x[1])
        print("====================================================================")
        print("human assert info:")
        print(f"num_human_bodies: {self.num_human_bodies}")
        print(f"human_asset_rigid_body_dict: {self.human_asset_rigid_body_dict}")
        print("--------------------------------------------------------------------")
        print(f"sorted_human_asset_rigid_body_dict: {self.sorted_human_asset_rigid_body_dict}")
        print("--------------------------------------------------------------------")
        print(f"num_human_shapes: {self.num_human_shapes}")
        print("--------------------------------------------------------------------")
        print(f"num_human_dofs: {self.num_human_dofs}")
        print(f"human_asset_dof_names: {self.human_asset_dof_names}")
        print("--------------------------------------------------------------------")
        print(f"num_human_actuators: {self.num_human_actuators}")
        print(f"human_asset_actuator_names: {self.human_asset_actuator_names}")
        print("--------------------------------------------------------------------")
        print(f"num_human_tendons: {self.num_human_tendons}")
        print("====================================================================")
        self.actuated_dof_indices = [i for i in range(self.num_human_dofs)]
        human_dof_props = self.gym.get_asset_dof_properties(human_asset)

        self.human_dof_lower_limits = []
        self.human_dof_upper_limits = []
        self.human_dof_default_pos = []
        self.human_dof_default_vel = []
        for i in range(self.num_human_dofs):
            human_dof_props['velocity'][i] = 3.14
            human_dof_props['effort'][i] = 1.0
            human_dof_props['friction'][i] = 0.1
            human_dof_props['stiffness'][i] = 1
            human_dof_props['armature'][i] = 0.1
            human_dof_props['damping'][i] = 0
            self.human_dof_lower_limits.append(human_dof_props['lower'][i])
            self.human_dof_upper_limits.append(human_dof_props['upper'][i])
            self.human_dof_default_pos.append(0.0)
            self.human_dof_default_vel.append(0.0)

        self.human_dof_lower_limits = to_torch(self.human_dof_lower_limits, device=self.device)
        self.human_dof_upper_limits = to_torch(self.human_dof_upper_limits, device=self.device)
        self.human_dof_default_pos = to_torch(self.human_dof_default_pos, device=self.device)
        self.human_dof_default_vel = to_torch(self.human_dof_default_vel, device=self.device)

        spacing = 1.5
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        human_start_pose = gymapi.Transform()
        human_start_pose.p = gymapi.Vec3(0, 0, 0.8)
        human_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        max_agg_bodies = self.num_human_bodies + 3
        max_agg_shapes = self.num_human_shapes + 3

        self.envs = []
        self.humans = []
        self.human_start_states = []
        self.human_indices = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, 1)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            human_actor = self.gym.create_actor(env_ptr, human_asset, human_start_pose, "human", 1, 1, 0)
            for dof_n in self.human_asset_dof_names:
                dof_id = self.gym.find_actor_dof_index(env_ptr, human_actor, dof_n, gymapi.DOMAIN_SIM)
                print(dof_n, dof_id)
            self.human_start_states.append(
                [human_start_pose.p.x, human_start_pose.p.y, human_start_pose.p.z,
                 human_start_pose.r.x, human_start_pose.r.y, human_start_pose.r.z,
                 human_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, human_actor, human_dof_props)
            human_idx = self.gym.get_actor_index(env_ptr, human_actor, gymapi.DOMAIN_SIM)
            self.human_indices.append(human_idx)
            human_props = self.gym.get_actor_rigid_shape_properties(env_ptr, human_actor)
            for p in human_props:
                p.friction = 10
            self.gym.set_actor_rigid_shape_properties(env_ptr, human_actor, human_props)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            self.humans.append(human_actor)

        self.human_start_states = to_torch(self.human_start_states, device=self.device).view(self.num_envs, 13)

    def create_sim(self):
        sim = self.gym.create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def set_sim_params_up_axis(self, sim_params, up_axis):
        if self.up_axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def render(self):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)

            self.gym.sync_frame_time(self.sim)

    def parse_sim_params(self):
        sim_params = gymapi.SimParams()
        sim_params.dt = 1. / 60.
        sim_params.substeps = 2
        sim_params.num_client_threads = 0

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = False
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.rest_offset = 0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 1000
        sim_params.physx.default_buffer_size_multiplier = 5.0
        sim_params.physx.num_subscenes = 0
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        sim_params.physx.contact_collection = gymapi.ContactCollection(2)
        sim_params.use_gpu_pipeline = False

        return sim_params


if __name__ == '__main__':
    '''
    ti -- 60 dof
    '''
    urdf_model = ['../assert/ti5robot/urdf', 'ti_xml.urdf']  # Root, Path

    Humanoid(urdf_model[0], urdf_model[1])