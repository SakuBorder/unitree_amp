from legged_gym.envs.g1.g1_amp_config import G1AMPCfg, G1AMPCfgPPO

class G1ClimbCfg(G1AMPCfg):
    """G1攀爬任务配置"""
    
    class env(G1AMPCfg.env):
        num_observations = G1AMPCfg.env.num_observations + 27  # 基础 + 任务观测
        num_privileged_obs = G1AMPCfg.env.num_privileged_obs + 27
        enableTaskObs = True

        climb_vel_penalty = True
        climb_vel_pen_coeff = 0.5
        climb_vel_pen_threshold = 0.2
        
        mode = "train"
        objCategories = ["stairs", "ramp", "obstacle"]
        
        enableIET = True  # 交互早期终止
        successThreshold = 0.5
        maxIETSteps = 100
        
    class rewards(G1AMPCfg.rewards):
        class scales(G1AMPCfg.rewards.scales):
            tracking_lin_vel = 0.5
            climb_progress = 2.0
            feet_clearance = 1.0
            climb_stability = 1.5

class G1ClimbCfgPPO(G1AMPCfgPPO):
    class runner(G1AMPCfgPPO.runner):
        experiment_name = "g1_climb"
        max_iterations = 20000
