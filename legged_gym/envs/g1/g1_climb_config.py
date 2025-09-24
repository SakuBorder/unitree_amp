from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO

class G1ClimbCfg(G1RoughCfg):
    """G1攀爬任务配置"""
    
    class env(G1RoughCfg.env):
        num_observations = 47 + 27  # 基础 + 任务观测
        enableTaskObs = True
        
        climb_vel_penalty = True
        climb_vel_pen_coeff = 0.5
        climb_vel_pen_threshold = 0.2
        
        mode = "train"
        objCategories = ["stairs", "ramp", "obstacle"]
        
        enableIET = True  # 交互早期终止
        successThreshold = 0.5
        maxIETSteps = 100
        
    class rewards(G1RoughCfg.rewards):
        class scales(G1RoughCfg.rewards.scales):
            tracking_lin_vel = 0.5
            climb_progress = 2.0
            feet_clearance = 1.0
            climb_stability = 1.5

class G1ClimbCfgPPO(G1RoughCfgPPO):
    class runner(G1RoughCfgPPO.runner):
        experiment_name = 'g1_climb'
        max_iterations = 20000