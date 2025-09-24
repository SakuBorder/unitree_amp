from legged_gym.envs.g1.g1_amp_config import G1AMPCfg, G1AMPCfgPPO

class G1TrajCfg(G1AMPCfg):
    """G1轨迹跟踪任务配置"""
    
    class env(G1AMPCfg.env):
        num_observations = G1AMPCfg.env.num_observations + 20  # 基础 + 轨迹点
        num_privileged_obs = G1AMPCfg.env.num_privileged_obs + 20
        enableTaskObs = True
        
    class traj:
        enable_markers = True  # 启用轨迹标记可视化
        num_samples = 10       # 轨迹采样点数
        sample_timestep = 0.5  # 采样间隔
        speed_min = 0.5
        speed_max = 2.0
        accel_max = 1.0
        sharp_turn_prob = 0.2
        sharp_turn_angle = 1.57  # 90度
        
    class rewards(G1AMPCfg.rewards):
        class scales(G1AMPCfg.rewards.scales):
            tracking_lin_vel = 0.5
            traj_tracking = 3.0
            traj_orientation = 1.0
            smooth_motion = 0.5

class G1TrajCfgPPO(G1AMPCfgPPO):
    class runner(G1AMPCfgPPO.runner):
        experiment_name = "g1_traj"
        max_iterations = 15000
