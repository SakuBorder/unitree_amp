from legged_gym.envs.g1.g1_amp_config import G1AMPCfg, G1AMPCfgPPO

class G1TrajCfg(G1AMPCfg):
    """G1轨迹跟踪任务配置"""
    
    class env(G1AMPCfg.env):
        num_observations = G1AMPCfg.env.num_observations + 20  # 基础 + 轨迹点
        num_privileged_obs = G1AMPCfg.env.num_privileged_obs + 20
        enableTaskObs = True
    class viewer:
        ref_env = 0
        pos = [10, 0, 3]  # [m]
        lookat = [0., 1, 0.]  # [m]
    class traj:
        enable_markers = True  # 启用轨迹标记可视化
        num_samples = 10       # 轨迹采样点数
        sample_timestep = 0.5  # 采样间隔
        speed_min = 0.5
        speed_max = 2.0
        accel_max = 1.0
        sharp_turn_prob = 0.2
        sharp_turn_angle = 1.57  # 90度
    class amp:
        # Number of stacked AMP observation steps.  A single step keeps the
        # implementation simple while remaining compatible with AMP training.
        num_obs_steps = 1
        # Whether to express AMP features in the local heading frame and
        # whether to include the root height explicitly.
        local_root_obs = True
        root_height_obs = True
        # Key bodies used for AMP features (typically the feet).
        key_body_names = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]   
    class rewards(G1AMPCfg.rewards):
        class scales(G1AMPCfg.rewards.scales):
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            traj_tracking = 1.0
            traj_orientation = 0.0
            smooth_motion = 0.0

class G1TrajCfgPPO(G1AMPCfgPPO):
    style_reward_weight = 0.5
    task_reward_weight = 0.5
    class algorithm(G1AMPCfgPPO.algorithm):
        learning_rate = 3.0e-4
    class runner(G1AMPCfgPPO.runner):
        debug_trajectory = False
        experiment_name = "g1_traj"
        max_iterations = 15000