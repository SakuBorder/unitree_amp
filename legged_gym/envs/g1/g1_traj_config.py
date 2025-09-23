import math

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO


class G1TrajCfg(G1RoughCfg):
    class traj:
        num_samples = 10
        sample_timestep = 0.5
        num_verts = 101
        dtheta_max = 2.0
        speed_min = 0.2
        speed_max = 1.0
        accel_max = 0.75
        sharp_turn_prob = 0.15
        sharp_turn_angle = math.pi / 2.0
        pos_error_scale = 2.0

    class env(G1RoughCfg.env):
        num_observations = G1RoughCfg.env.num_observations + 2 * traj.num_samples
        num_privileged_obs = G1RoughCfg.env.num_privileged_obs + 2 * traj.num_samples

    class rewards(G1RoughCfg.rewards):
        class scales(G1RoughCfg.rewards.scales):
            traj_tracking = 1.0


class G1TrajCfgPPO(G1RoughCfgPPO):
    class runner(G1RoughCfgPPO.runner):
        experiment_name = "g1_traj"
        run_name = ""
