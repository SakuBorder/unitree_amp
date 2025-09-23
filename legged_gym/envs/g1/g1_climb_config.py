from legged_gym.envs.g1.g1_amp_config import G1AMPCfg, G1AMPCfgPPO


class G1ClimbCfg(G1AMPCfg):
    class climb:
        distance_range = (1.5, 3.0)
        lateral_range = 0.6
        height_range = (0.6, 1.2)
        height_error_scale = 6.0

    class env(G1AMPCfg.env):
        pass

    class rewards(G1AMPCfg.rewards):
        class scales(G1AMPCfg.rewards.scales):
            climb_progress = 1.0
            climb_height = 0.6
            climb_upright = 0.4
            climb_slip = -0.2


class G1ClimbCfgPPO(G1AMPCfgPPO):
    class runner(G1AMPCfgPPO.runner):
        experiment_name = "g1_climb"
        run_name = ""


G1ClimbCfg.env.num_observations = G1AMPCfg.env.num_observations + 3
G1ClimbCfg.env.num_privileged_obs = G1AMPCfg.env.num_privileged_obs + 3
