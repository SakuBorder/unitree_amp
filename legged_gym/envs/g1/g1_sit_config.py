from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO

class G1SitCfg(G1RoughCfg):
    """G1坐下任务配置"""
    
    class env(G1RoughCfg.env):
        num_observations = 47 + 37  # 基础 + 任务观测
        enableTaskObs = True
        enableConditionalDisc = False
        
        sit_vel_penalty = True
        sit_vel_pen_coeff = 0.5
        sit_vel_pen_threshold = 0.2
        sit_ang_vel_pen_coeff = 0.1
        sit_ang_vel_pen_threshold = 0.1
        
        mode = "train"
        objCategories = ["chair", "stool", "bench"]
        
        class eval:
            successThreshold = 0.3
            
    class rewards(G1RoughCfg.rewards):
        class scales(G1RoughCfg.rewards.scales):
            tracking_lin_vel = 0.3
            sit_position = 2.0
            sit_orientation = 1.0
            stable_sitting = 1.5

class G1SitCfgPPO(G1RoughCfgPPO):
    class runner(G1RoughCfgPPO.runner):
        experiment_name = 'g1_sit'
        max_iterations = 15000