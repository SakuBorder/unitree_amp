"""Configuration for the G1 AMP task."""

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO


class G1AMPCfg(G1RoughCfg):
    """Environment configuration for the G1 AMP locomotion task."""

    class env(G1RoughCfg.env):
        # AMP shares the same proprioceptive observation space as the standard
        # locomotion task.
        num_observations = G1RoughCfg.env.num_observations
        num_privileged_obs = G1RoughCfg.env.num_privileged_obs

    class amp:
        # Number of stacked AMP observation steps.  A single step keeps the
        # implementation simple while remaining compatible with AMP training.
        num_obs_steps = 1
        # Whether to express AMP features in the local heading frame and
        # whether to include the root height explicitly.
        local_root_obs = True
        root_height_obs = True
        # Key bodies used for AMP features (typically the feet).
        key_body_names = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]

    class observations:
        class amp:
            class joint_pos:
                class params:
                    class asset_cfg:
                        joint_names = [
                            "left_hip_yaw_joint",
                            "left_hip_roll_joint",
                            "left_hip_pitch_joint",
                            "left_knee_joint",
                            "left_ankle_pitch_joint",
                            "left_ankle_roll_joint",
                            "right_hip_yaw_joint",
                            "right_hip_roll_joint",
                            "right_hip_pitch_joint",
                            "right_knee_joint",
                            "right_ankle_pitch_joint",
                            "right_ankle_roll_joint",
                            "torso_joint",
                        ]

            class feet:
                class params:
                    class asset_cfg:
                        body_names = [
                            "left_ankle_roll_link",
                            "right_ankle_roll_link",
                        ]

    class motion_lib:
        # Path to the motion library used for AMP demonstrations.  Users
        # should replace this with the path to their processed motion dataset.
        motion_file = "/home/dy/dy/code/unitree_amp/data/g1_12/walk"
        # MJCF description used to build the skeleton that matches the motion
        # library.  The default points to the bundled G1 description.
        mjcf_file = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.xml"
        # Number of motion clips to load per batch when initialising the
        # library.  Loading every clip can be prohibitively expensive, so a
        # smaller subset keeps start-up times manageable.
        num_motions_per_batch = 256
        # Multi-threaded motion loading improves throughput when many clips are
        # requested.
        multi_thread = True
        # G1 motions already include the desired height, so no additional
        # height correction is required.
        fix_height_mode = "no_fix"


class G1AMPCfgPPO(G1RoughCfgPPO):
    """Training configuration for AMP + PPO on the G1 task."""

    # Use the specialised AMP on-policy runner instead of the default PPO
    # runner provided by rsl_rl.
    runner_class_name = "AMPOnPolicyRunner"
    # Balance task tracking rewards and AMP style rewards.
    style_reward_weight = 0.5
    task_reward_weight = 0.5
    # AMP observations are pre-normalised, so empirical normalisation is not
    # required and would add extra overhead.
    empirical_normalization = False

    class policy(G1RoughCfgPPO.policy):
        # Explicitly state the policy network class used by the AMP runner.
        class_name = "ActorCritic"
        actor_hidden_dims = [256, 256]
        critic_hidden_dims = [256, 256]

    class algorithm(G1RoughCfgPPO.algorithm):
        # AMP integrates with PPO through the AMP_PPO algorithm.
        class_name = "AMP_PPO"
        learning_rate = 3.0e-4
        num_learning_epochs = 4
        num_mini_batches = 4

    class discriminator:
        # Discriminator architecture closely follows common AMP practices.
        hidden_dims = [256, 256]
        reward_scale = 1.0
        loss_type = "BCEWithLogits"
        disc_logit_reg = 0.01
        disc_grad_penalty = 10.0
        disc_weight_decay = 0.0

    class runner(G1RoughCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "AMP_PPO"
        num_steps_per_env = 24
        max_iterations = 10000
        experiment_name = "g1_amp"
