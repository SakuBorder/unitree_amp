from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO

class G1CarryCfg(G1RoughCfg):
    """G1搬运任务配置"""
    
    class env(G1RoughCfg.env):
        num_observations = 47 + 42  # 基础观测 + 任务观测
        num_privileged_obs = 50 + 42
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
            
    class rewards(G1RoughCfg.rewards):
        class scales(G1RoughCfg.rewards.scales):
            # 保留原有奖励
            tracking_lin_vel = 0.5  # 降低权重
            tracking_ang_vel = 0.25
            # 新增搬运任务奖励
            carry_walk = 1.0
            carry_vel = 1.0
            handheld = 0.5
            putdown = 1.0
            box_orientation = -1.0

class G1CarryCfgPPO(G1RoughCfgPPO):
    class runner(G1RoughCfgPPO.runner):
        experiment_name = 'g1_carry'
        max_iterations = 20000