from legged_gym.envs.g1.g1_amp_config import G1AMPCfg, G1AMPCfgPPO


class G1SitCfg(G1AMPCfg):
    class sit:
        target_height = 0.45
        target_radius = 0.8
        min_radius = 0.2
        height_noise = 0.05
        pos_error_scale = 4.0
        height_error_scale = 8.0
        velocity_penalty_scale = 2.0
        ang_vel_penalty_scale = 0.5

    class motion_lib(G1AMPCfg.motion_lib):
        sampling_groups = (
            {"name": "sit", "contains": ("sit",), "weight": 0.7},
            {"name": "loco", "contains": ("loco", "walk"), "weight": 0.3},
        )
        default_sampling_weight = 0.0

    class env(G1AMPCfg.env):
        pass

    class rewards(G1AMPCfg.rewards):
        class scales(G1AMPCfg.rewards.scales):
            sit_position = 1.0
            sit_height = 0.5
            sit_still = 0.2
            sit_orientation = 0.3


class G1SitCfgPPO(G1AMPCfgPPO):
    class runner(G1AMPCfgPPO.runner):
        experiment_name = "g1_sit"
        run_name = ""


G1SitCfg.env.num_observations = G1AMPCfg.env.num_observations + 3
G1SitCfg.env.num_privileged_obs = G1AMPCfg.env.num_privileged_obs + 3
