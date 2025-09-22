import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import os
import shutil

from torch.utils.tensorboard import SummaryWriter


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # TensorBoard
    log_dir = config.get("log_dir", "debug/mujoco")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # 当前关节角（只取你控制的12个自由度）
            qj_now = d.qpos[7:7+num_actions]
            dqj_now = d.qvel[6:6+num_actions]

            tau = pd_control(target_dof_pos, qj_now, kps, np.zeros_like(kds), dqj_now, kds)
            d.ctrl[:] = tau

            # Step physics
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # 构造观测
                qj = (qj_now - default_angles) * dof_pos_scale
                dqj = dqj_now * dof_vel_scale
                quat = d.qpos[3:7]
                omega = d.qvel[3:6] * ang_vel_scale
                gravity_orientation = get_gravity_orientation(quat)

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase], dtype=np.float32)

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)

                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()

                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

                # === TensorBoard Logging ===
                step = counter // control_decimation

                # 1) target pos vs 当前 qj（每关节一张图，两条曲线）
                for j in range(num_actions):
                    writer.add_scalars(
                        f'fig_mujoco/target_vs_qj/j{j}',
                        {
                            'target_pos': float(target_dof_pos[j]),
                            'qj': float(qj_now[j])
                        },
                        step
                    )

                # 2) action 左右对比（0-5 vs 6-11）
                for j in range(num_actions // 2):
                    if j + num_actions // 2 < num_actions:
                        writer.add_scalars(
                            f'fig_mujoco/action_{j}_vs_{j + num_actions//2}',
                            {
                                f'action_{j}': float(action[j]),
                                f'action_{j + num_actions//2}': float(action[j + num_actions//2])
                            },
                            step
                        )

                # 3) dof 左右对比（当前 qj）
                for j in range(num_actions // 2):
                    if j + num_actions // 2 < num_actions:
                        writer.add_scalars(
                            f'fig_mujoco/dof_{j}_vs_{j + num_actions//2}',
                            {
                                f'dof_{j}': float(qj_now[j]),
                                f'dof_{j + num_actions//2}': float(qj_now[j + num_actions//2])
                            },
                            step
                        )

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
