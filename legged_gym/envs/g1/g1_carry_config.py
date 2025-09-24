from legged_gym.envs.g1.g1_amp_config import G1AMPCfg, G1AMPCfgPPO

class G1CarryCfg(G1AMPCfg):
    """G1搬运任务配置"""
    
    class env(G1AMPCfg.env):
        num_observations = G1AMPCfg.env.num_observations + 42  # 基础观测 + 任务观测
        num_privileged_obs = G1AMPCfg.env.num_privileged_obs + 42
        enableTaskObs = True
        onlyVelReward = False
        onlyHeightHandHeldReward = False
        
        box_vel_penalty = True
        box_vel_pen_coeff = 0.5
        box_vel_pen_threshold = 0.3
        
        mode = "train"  # train or test
        
    class box:
        class build:
            baseSize = [0.3, 0.3, 0.3]  # 箱子基础尺寸
            randomSize = True
            randomModeEqualProportion = True
            scaleRangeX = [0.8, 1.2]
            scaleRangeY = [0.8, 1.2]
            scaleRangeZ = [0.8, 1.2]
            scaleSampleInterval = 0.1
            randomDensity = True
            testSizes = [[0.3, 0.3, 0.3], [0.4, 0.4, 0.4]]
            
        class reset:
            randomRot = True
            randomHeight = False
            randomHeightProb = 0.0
            maxTopSurfaceHeight = 1.5
            
        class obs:
            enableBboxObs = True
            
    class rewards(G1AMPCfg.rewards):
        class scales(G1AMPCfg.rewards.scales):
            # 保留原有奖励
            tracking_lin_vel = 0.5  # 降低权重
            tracking_ang_vel = 0.25
            # 新增搬运任务奖励
            carry_walk = 1.0
            carry_vel = 1.0
            handheld = 0.5
            putdown = 1.0
            box_orientation = -1.0

class G1CarryCfgPPO(G1AMPCfgPPO):
    class runner(G1AMPCfgPPO.runner):
        experiment_name = "g1_carry"
        max_iterations = 20000