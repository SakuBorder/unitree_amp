from legged_gym.envs.g1.g1_amp_config import G1AMPCfg, G1AMPCfgPPO

class G1SitCfg(G1AMPCfg):
    """G1坐下任务配置"""
    
    class env(G1AMPCfg.env):
        num_observations = G1AMPCfg.env.num_observations + 39  # 基础 + 任务观测
        num_privileged_obs = G1AMPCfg.env.num_privileged_obs + 39
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
            
    class rewards(G1AMPCfg.rewards):
        class scales(G1AMPCfg.rewards.scales):
            tracking_lin_vel = 0.3
            sit_position = 2.0
            sit_orientation = 1.0
            stable_sitting = 1.5

class G1SitCfgPPO(G1AMPCfgPPO):
    class runner(G1AMPCfgPPO.runner):
        experiment_name = "g1_sit"
        max_iterations = 15000