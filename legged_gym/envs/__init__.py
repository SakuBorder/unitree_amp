from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1.h1_env import H1Robot
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.g1.g1_amp_config import G1AMPCfg, G1AMPCfgPPO
from legged_gym.envs.g1.g1_amp_env import G1AMPRobot
# from legged_gym.envs.g1.g1_traj_config import G1TrajCfg, G1TrajCfgPPO
# from legged_gym.envs.g1.g1_traj_env import G1TrajRobot
# from legged_gym.envs.g1.g1_sit_config import G1SitCfg, G1SitCfgPPO
# from legged_gym.envs.g1.g1_sit_env import G1SitRobot
# from legged_gym.envs.g1.g1_carry_config import G1CarryCfg, G1CarryCfgPPO
# from legged_gym.envs.g1.g1_carry_env import G1CarryRobot
# from legged_gym.envs.g1.g1_climb_config import G1ClimbCfg, G1ClimbCfgPPO
# from legged_gym.envs.g1.g1_climb_env import G1ClimbRobot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register( "g1_amp", G1AMPRobot, G1AMPCfg(), G1AMPCfgPPO())
# task_registry.register( "g1_traj", G1TrajRobot, G1TrajCfg(), G1TrajCfgPPO())
# task_registry.register( "g1_sit", G1SitRobot, G1SitCfg(), G1SitCfgPPO())
# task_registry.register( "g1_carry", G1CarryRobot, G1CarryCfg(), G1CarryCfgPPO())
# task_registry.register( "g1_climb", G1ClimbRobot, G1ClimbCfg(), G1ClimbCfgPPO())