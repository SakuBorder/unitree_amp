import math

from legged_gym.envs.g1.g1_amp_config import G1AMPCfg, G1AMPCfgPPO


class G1TrajCfg(G1AMPCfg):
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
        enable_debug_vis = True
        debug_height = 0.05
        sample_marker_height = 0.15

    class env(G1AMPCfg.env):
        pass

    class rewards(G1AMPCfg.rewards):
        class scales(G1AMPCfg.rewards.scales):
            traj_tracking = 1.0


class G1TrajCfgPPO(G1AMPCfgPPO):
    class runner(G1AMPCfgPPO.runner):
        experiment_name = "g1_traj"
        run_name = ""


# Update the environment observation dimensions to include trajectory features.
G1TrajCfg.env.num_observations = (
    G1AMPCfg.env.num_observations + 2 * G1TrajCfg.traj.num_samples
)
G1TrajCfg.env.num_privileged_obs = (
    G1AMPCfg.env.num_privileged_obs + 2 * G1TrajCfg.traj.num_samples
)
