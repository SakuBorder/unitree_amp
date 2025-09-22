# vis_taihu.py

import numpy as np
import os
import os.path as osp

from isaacgym import gymapi, gymutil, gymtorch

import torch
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree

# 确保能找到 MotionLibTaihu
import sys
sys.path.append(os.getcwd())
from phc.phc.utils.motion_lib_taihu import MotionLibTaihu
from phc.phc.utils.flags import flags

# ---------------- 基本开关 ----------------
flags.test = True
flags.im_eval = True

args = gymutil.parse_arguments(description="Robot Motion Visualizer")

# ---------------- Gym 初始化 ----------------
gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.001

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# 地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.restitution = 0.0
gym.add_ground(sim, plane_params)

# 观察器
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()
cam_pos = gymapi.Vec3(3.0, 3.0, 2.0)
cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# ---------------- 载入资产 ----------------
asset_root = "./"
asset_file = "assert/ti5/tai5_12dof_no_feet.urdf"
g1_xml = "assert/ti5/ti_no_limit_12.xml"

print(f"Loading asset: {asset_file}")
if not osp.exists(asset_file):
    print(f"*** Asset file not found: {asset_file}")
    quit()

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.use_mesh_materials = True
asset_options.vhacd_enabled = False
asset_options.disable_gravity = False
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

try:
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    print("Asset loaded successfully")
except Exception as e:
    print(f"*** Failed to load asset: {e}")
    quit()

num_dofs = gym.get_asset_dof_count(asset)
num_bodies = gym.get_asset_rigid_body_count(asset)
print(f"Asset info: {num_dofs} DOFs, {num_bodies} bodies")

# 创建环境与 actor
env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)
env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
env = gym.create_env(sim, env_lower, env_upper, 1)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

actor_handle = gym.create_actor(env, asset, pose, "robot", 0, 1)

# DOF 参数
dof_props = gym.get_actor_dof_properties(env, actor_handle)
for i in range(num_dofs):
    dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    dof_props['stiffness'][i] = 300.0
    dof_props['damping'][i] = 10.0
    dof_props['velocity'][i] = 100.0
    dof_props['effort'][i] = 100.0
gym.set_actor_dof_properties(env, actor_handle, dof_props)

# ---------------- 站立姿态 ----------------
def get_standing_pose():
    pose = np.zeros(num_dofs)
    joint_names = [gym.get_asset_dof_name(asset, i) for i in range(num_dofs)]
    for i, name in enumerate(joint_names):
        if 'KNEE_P' in name:
            pose[i] = 0.3
        elif 'ANKLE_P' in name:
            pose[i] = -0.15
    return pose

standing_pose = get_standing_pose()

dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
for i in range(num_dofs):
    dof_states['pos'][i] = standing_pose[i]
    dof_states['vel'][i] = 0.0
gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

# ---------------- 载入 MotionLib（开关可从外部控制） ----------------
print("Loading motion library...")
motion_file = "data/ti512/v1/walk/0-Male2Walking_c3d_B15 -  Walk turn around_poses.pkl"

motion_lib = None
sk_tree = None
if osp.exists(motion_file):
    try:
        sim_device = torch.device(args.sim_device)

        urdf_dof_names = [gym.get_asset_dof_name(asset, i) for i in range(num_dofs)]
        print(f"URDF DOF names: {urdf_dof_names}")

        sk_tree = SkeletonTree.from_mjcf(g1_xml)

        # 这里直接传入 use_joint_mapping / debug_dof
        motion_lib = MotionLibTaihu(
            motion_file=motion_file,
            device=sim_device,
            masterfoot_conifg=None,
            fix_height=False,
            multi_thread=False,
            mjcf_file=g1_xml,
            use_joint_mapping=True,   # 外部初始选择
            debug_dof=True,            # 外部初始选择
        )

        print("\n=== 测试 Joint Mapping 开关效果 ===")
        print(f"初始 mapping 设置: {motion_lib.use_joint_mapping}")

        # 先按当前设置加载一次
        num_motions = 1
        motion_lib.load_motions(
            skeleton_trees=[sk_tree] * num_motions,
            gender_betas=[torch.zeros(17)] * num_motions,
            limb_weights=[np.zeros(10)] * num_motions,
            random_sample=True
        )
        print(f"Motion library loaded successfully with {num_motions} motions")
        print(f"Motion keys: {getattr(motion_lib, 'curr_motion_keys', None)}")

        # 对比：关/开映射
        test_motion_id = torch.tensor([0], device=sim_device)
        test_time = torch.tensor([0.0], device=sim_device)

        motion_lib.set_joint_mapping_mode(use_mapping=False, debug=True)
        res_no_map = motion_lib.get_motion_state(test_motion_id, test_time)

        motion_lib.set_joint_mapping_mode(use_mapping=True, debug=True)
        res_map = motion_lib.get_motion_state(test_motion_id, test_time)

        print(f"无 mapping 的 DOF 前5: {res_no_map['dof_pos'][0, :5]}")
        print(f"有 mapping 的 DOF 前5: {res_map['dof_pos'][0, :5]}")
        print("=== 对比测试结束 ===\n")

        # 设定最终运行时希望的模式（可按需改）
        motion_lib.set_joint_mapping_mode(use_mapping=False, debug=True)

    except Exception as e:
        print(f"Failed to load motion library: {e}")
        import traceback
        traceback.print_exc()
        motion_lib = None
        sk_tree = None
else:
    print(f"*** Motion file not found: {motion_file}")
    print("Use standing pose only")

print("Preparing simulation...")
gym.prepare_sim(sim)

num_envs = 1
env_ids = torch.arange(num_envs, dtype=torch.int32, device=args.sim_device)

current_pose = torch.tensor(standing_pose, dtype=torch.float32, device=args.sim_device).unsqueeze(0)

dof_states_tensor = gym.acquire_dof_state_tensor(sim)
dof_states_tensor = gymtorch.wrap_tensor(dof_states_tensor)

rigidbody_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
rigidbody_state_tensor = gymtorch.wrap_tensor(rigidbody_state_tensor)
rigidbody_state_tensor = rigidbody_state_tensor.reshape(num_envs, -1, 13)

actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
actor_root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor)

try:
    contact_force_tensor = gym.acquire_net_contact_force_tensor(sim)
    contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
    contact_force_tensor = contact_force_tensor.view(num_envs, -1, 3)
    contact_available = True
    print("Contact force tensor acquired successfully")
except Exception as e:
    print(f"Warning: Could not acquire contact force tensor: {e}")
    contact_available = False

body_names = [gym.get_asset_rigid_body_name(asset, i) for i in range(num_bodies)]
print(f"Rigid bodies: {body_names}")

# ---------------- 键盘控制（J 键切换映射） ----------------
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "toggle_pause")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_M, "toggle_motion")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_J, "toggle_joint_mapping")

print("Simulation ready!")
print("Controls:")
print("  ESC   - Exit")
print("  SPACE - Pause/Resume")
print("  M     - Toggle Motion/Standing")
print("  R     - Reset motion")
print("  J     - Toggle Joint Mapping (debug)")

# ---------------- 主循环 ----------------
paused = False
use_motion = motion_lib is not None
motion_time = 0.0
motion_id = 0
dt = sim_params.dt

print("Starting visualization...")
print("Motion playback " + ("enabled" if use_motion else "DISABLED (standing pose only)"))

frame_count = 0

while not gym.query_viewer_has_closed(viewer):
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "toggle_pause" and evt.value > 0:
            paused = not paused
            print(f"Simulation {'PAUSED' if paused else 'RESUMED'}")
        elif evt.action == "toggle_motion" and evt.value > 0 and motion_lib is not None:
            use_motion = not use_motion
            motion_time = 0.0
            print(f"Motion {'ENABLED' if use_motion else 'DISABLED'}")
        elif evt.action == "reset" and evt.value > 0:
            motion_time = 0.0
            print("Motion reset")
        elif evt.action == "toggle_joint_mapping" and evt.value > 0 and motion_lib is not None:
            curr = motion_lib.use_joint_mapping
            motion_lib.set_joint_mapping_mode(use_mapping=not curr, debug=True)
            print(f"Joint mapping switched: {curr} -> {not curr}")

    if not paused:
        if use_motion and motion_lib is not None:
            try:
                motion_len = motion_lib.get_motion_length(motion_id).item()
                motion_time_wrapped = motion_time % motion_len

                motion_res = motion_lib.get_motion_state(
                    torch.tensor([motion_id], device=args.sim_device),
                    torch.tensor([motion_time_wrapped], device=args.sim_device)
                )

                root_pos = motion_res["root_pos"]
                root_rot = motion_res["root_rot"]
                dof_pos  = motion_res["dof_pos"]
                root_vel = motion_res["root_vel"]
                root_ang_vel = motion_res["root_ang_vel"]

                if frame_count % 60 == 0:
                    print(f"Frame {frame_count}: t={motion_time_wrapped:.2f}s, DOF shape={dof_pos.shape}")
                    print(f"DOF[0:5]: {dof_pos[0, :5]} (mapping={'ON' if motion_lib.use_joint_mapping else 'OFF'})")

                dof_pos_clamped = torch.clamp(dof_pos, -3.14, 3.14)
                zeros_vel = torch.zeros_like(dof_pos_clamped)
                dof_state = torch.stack([dof_pos_clamped, zeros_vel], dim=-1).repeat(num_envs, 1, 1).contiguous()

                root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1).repeat(num_envs, 1).contiguous()

                gym.set_actor_root_state_tensor_indexed(
                    sim, gymtorch.unwrap_tensor(root_states),
                    gymtorch.unwrap_tensor(env_ids), len(env_ids)
                )
                gym.set_dof_state_tensor_indexed(
                    sim, gymtorch.unwrap_tensor(dof_state),
                    gymtorch.unwrap_tensor(env_ids), len(env_ids)
                )

                motion_time += dt

            except Exception as e:
                print(f"Motion playback error: {e}")
                import traceback
                traceback.print_exc()
                use_motion = False
        else:
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(current_pose))

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)

        if contact_available:
            try:
                gym.refresh_net_contact_force_tensor(sim)
            except Exception:
                pass

        frame_count += 1

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

print("Cleaning up...")
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("Done!")
