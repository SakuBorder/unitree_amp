import math

from legged_gym.envs.g1.g1_amp_config import G1AMPCfg, G1AMPCfgPPO


class G1TrajCfg(G1AMPCfg):
    """G1 轨迹跟随任务配置（在 AMP 基础上增加轨迹相关超参，并扩展观测维度）。"""

    class traj:
        # 轨迹采样/生成参数
        num_samples = 10
        sample_timestep = 0.5
        num_verts = 101
        dtheta_max = 2.0
        speed_min = 0.2
        speed_max = 1.0
        accel_max = 0.75
        sharp_turn_prob = 0.15
        sharp_turn_angle = math.pi / 2.0
        pos_error_scale = 2.0  # 用于 compute_traj_reward 的系数
        # 调试绘制（仅画线，不涉及 marker actor）
        enable_debug_vis = True
        debug_height = 0.05
        sample_marker_height = 0.15  # 若后续想用 marker，可复用此高度

    class env(G1AMPCfg.env):
        # 保持与 AMP 相同的环境配置（不改动）
        pass

    class motion_lib(G1AMPCfg.motion_lib):
        # 减少一次批量加载的 motion 数量，避免初始化过慢或占用过大
        num_motions_per_batch = 16

    class rewards(G1AMPCfg.rewards):
        class scales(G1AMPCfg.rewards.scales):
            # 轨迹跟随奖励；同时关闭常规速度跟踪（由轨迹主导）
            traj_tracking = 1.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0


class G1TrajCfgPPO(G1AMPCfgPPO):
    """PPO 训练配置（沿用 AMP 的 AMP_PPO 流程，仅更改实验名等）。"""

    class runner(G1AMPCfgPPO.runner):
        experiment_name = "g1_traj"
        run_name = ""
        num_steps_per_env = 24
        max_iterations = 50_000


# === 扩展观测维度：每个样本提供 2 维 (x,y) 的局部轨迹点 ===
G1TrajCfg.env.num_observations = (
    G1AMPCfg.env.num_observations + 2 * G1TrajCfg.traj.num_samples
)
G1TrajCfg.env.num_privileged_obs = (
    G1AMPCfg.env.num_privileged_obs + 2 * G1TrajCfg.traj.num_samples
)
