from legged_gym.envs.g1.g1_traj_config import G1TrajCfg, G1TrajCfgPPO


class G1CarryCfg(G1TrajCfg):
    class carry:
        height_target = 0.9
        height_noise = 0.05
        height_error_scale = 6.0
        ang_vel_penalty_scale = 1.5
        lateral_vel_penalty_scale = 1.0

    class motion_lib(G1TrajCfg.motion_lib):
        sampling_groups = (
            {"name": "carryWith", "contains": ("carrywith",), "weight": 0.6},
            {"name": "pickUp", "contains": ("pickup",), "weight": 0.25},
            {"name": "loco", "contains": ("loco", "walk"), "weight": 0.15},
        )
        default_sampling_weight = 0.0

    class rewards(G1TrajCfg.rewards):
        class scales(G1TrajCfg.rewards.scales):
            carry_height = 0.75
            carry_upright = 0.5
            carry_stability = 0.3
            carry_lateral_penalty = -0.1


class G1CarryCfgPPO(G1TrajCfgPPO):
    class runner(G1TrajCfgPPO.runner):
        experiment_name = "g1_carry"
        run_name = ""


G1CarryCfg.env.num_observations = G1TrajCfg.env.num_observations + 2
G1CarryCfg.env.num_privileged_obs = G1TrajCfg.env.num_privileged_obs + 2