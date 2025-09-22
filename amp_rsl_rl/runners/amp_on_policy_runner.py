# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import statistics
import time
from collections import deque
from copy import deepcopy

import torch
from torch.onnx import errors as onnx_errors
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
import rsl_rl.utils
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl.utils import store_code_state

from amp_rsl_rl.utils import Normalizer
from amp_rsl_rl.algorithms import AMP_PPO
from amp_rsl_rl.networks import Discriminator, ActorCriticMoE
from amp_rsl_rl.utils import export_policy_as_onnx


class _MotionLibDataset:
    """Sample AMP expert transitions directly from an environment's MotionLib."""

    def __init__(self, env: VecEnv, delta_t: float) -> None:
        self.env = env
        self.delta_t = delta_t
        self.device = env.device
        # ``AMPOnPolicyRunner`` checks ``all_obs`` for diagnostics, so keep a
        # dummy tensor matching the environment's AMP observation dimension.
        self.all_obs = torch.zeros(1, 0, device=self.device)

    def pad_observations(self, target_dim: int) -> None:
        """Ensure internal buffers match ``target_dim`` features."""
        self.all_obs = torch.zeros(1, target_dim, device=self.device)

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """Yield mini-batches of (obs, next_obs) pairs from the motion library."""
        for _ in range(num_mini_batch):
            if getattr(self.env, "fetch_amp_obs_demo", None) is None:
                raise RuntimeError(
                    "Environment missing fetch_amp_obs_demo; cannot sample expert transitions"
                )

            obs, next_obs = self.env.fetch_amp_obs_demo(mini_batch_size, self.delta_t)
            yield obs, next_obs


class AMPOnPolicyRunner:
    """Trainer that combines on-policy RL with AMP using MotionLib demos.

    The environment must expose ``fetch_amp_obs_demo`` and an initialized
    MotionLib. Expert transitions are sampled directly from it without any
    disk-based loaders.
    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        # ``train_cfg`` follows the same structure expected by the base
        # ``OnPolicyRunner``.  The runner-specific parameters live under the
        # ``runner`` key, while policy/algorithm settings are grouped in their
        # respective sections.  Fall back gracefully if an older flat layout is
        # provided.
        self.train_cfg = train_cfg
        self.cfg = train_cfg.get("runner", train_cfg)
        self.alg_cfg = deepcopy(train_cfg.get("algorithm", {}))
        self.policy_cfg = deepcopy(train_cfg.get("policy", {}))
        self.discriminator_cfg = deepcopy(train_cfg.get("discriminator", {}))
        self.task_reward_weight = train_cfg.get("task_reward_weight", 0.5)
        self.style_reward_weight = train_cfg.get("style_reward_weight", 0.5)
        self.device = device
        self.env = env

        # Get the size of the observation space
        obs = self.env.get_observations()
        obs_extras = self._require_observation_extras()
        num_obs = obs.shape[1]
        critic_tensor = obs_extras.get("critic")
        num_critic_obs = critic_tensor.shape[1] if critic_tensor is not None else num_obs
        policy_class_name = self.cfg.get(
            "policy_class_name", self.policy_cfg.get("class_name", "ActorCritic")
        )
        self.policy_cfg.pop("class_name", None)
        actor_critic_class = eval(policy_class_name)  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticMoE = (
            actor_critic_class(
                num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
            ).to(self.device)
        )
        # NOTE: to use this we need to configure the observations in the env coherently with amp observation. Tested with Manager Based envs in Isaaclab
        amp_joint_names = self.env.cfg.observations.amp.joint_pos.params.asset_cfg.joint_names
        amp_foot_body_names = None
        if hasattr(self.env.cfg.observations.amp, "feet"):
            amp_foot_body_names = (
                self.env.cfg.observations.amp.feet.params.asset_cfg.body_names
            )

        # Older legged-gym configs store the control decimation under
        # `cfg.control.decimation` instead of the top-level `cfg.decimation`.
        # Fall back to that path if the direct attribute is missing so the
        # runner works across both styles of configuration.
        decimation = getattr(self.env.cfg, "decimation", None)
        if decimation is None:
            decimation = getattr(getattr(self.env.cfg, "control", object()), "decimation", 1)
        delta_t = self.env.cfg.sim.dt * decimation

        # Initialize all the ingredients required for AMP (discriminator, dataset loader).
        # Depending on the environment implementation, AMP observations might be
        # lazily populated only after the first physics step.  If the buffer is
        # still empty here, trigger a dummy post-physics step to populate it so
        # the normalizer is created with the correct dimensionality.
        num_amp_obs = obs_extras["amp"].shape[1]
        if num_amp_obs == 0:
            self.env.post_physics_step()
            obs_extras = self._require_observation_extras()
            num_amp_obs = obs_extras["amp"].shape[1]

        if not hasattr(self.env, "fetch_amp_obs_demo") or getattr(self.env, "_motion_lib", None) is None:
            raise RuntimeError("Environment must provide MotionLib demos via fetch_amp_obs_demo")
        amp_data = _MotionLibDataset(self.env, delta_t)
        try:
            amp_data.pad_observations(num_amp_obs)
        except ValueError as exc:
            raise ValueError(
                f"AMP dataset observation size {amp_data.all_obs.shape[1]} "
                f"!= environment size {num_amp_obs}"
            ) from exc

        # self.env.unwrapped.scene["robot"].joint_names)

        self.amp_normalizer = Normalizer(num_amp_obs, device=self.device)
        self.discriminator = Discriminator(
            input_dim=num_amp_obs * 2,  
            hidden_layer_sizes=self.discriminator_cfg["hidden_dims"],
            reward_scale=self.discriminator_cfg["reward_scale"],
            device=self.device,
            loss_type=self.discriminator_cfg.get("loss_type", "BCEWithLogits"),
            # TokenHSI alignment parameters
            disc_logit_reg=self.discriminator_cfg.get("disc_logit_reg", 0.01),
            disc_grad_penalty=self.discriminator_cfg.get("disc_grad_penalty", 10.0),
            disc_weight_decay=self.discriminator_cfg.get("disc_weight_decay", 0.0),
        ).to(self.device)

        # Initialize the PPO algorithm
        algorithm_class_name = self.cfg.get(
            "algorithm_class_name", self.alg_cfg.get("class_name", "AMP_PPO")
        )
        self.alg_cfg.pop("class_name", None)
        alg_class = eval(algorithm_class_name)  # AMP_PPO
        
        # Extract normalize_amp_input before processing alg_cfg to avoid conflicts
        normalize_amp_input = self.alg_cfg.pop("normalize_amp_input", True)
        
        # This removes from alg_cfg fields that are not in AMP_PPO but are introduced in rsl_rl 2.2.3 PPO
        # Also remove any parameters that might cause conflicts
        conflicting_params = [
            "normalize_advantage_per_mini_batch",
            "rnd_cfg", 
            "symmetry_cfg", 
            "multi_gpu_cfg",
            "disc_learning_rate",  # This might be in config but not in AMP_PPO
        ]
        
        for key in list(self.alg_cfg.keys()):
            if key not in AMP_PPO.__init__.__code__.co_varnames or key in conflicting_params:
                self.alg_cfg.pop(key)

        self.alg: AMP_PPO = alg_class(
            actor_critic=actor_critic,
            discriminator=self.discriminator,
            amp_data=amp_data,
            amp_normalizer=self.amp_normalizer,
            normalize_amp_input=normalize_amp_input,
            device=self.device,
            **self.alg_cfg,
        )
        
        self.num_steps_per_env = self.cfg.get(
            "num_steps_per_env", train_cfg.get("num_steps_per_env")
        )
        self.save_interval = self.cfg.get(
            "save_interval", train_cfg.get("save_interval", 0)
        )
        self.empirical_normalization = train_cfg.get("empirical_normalization", False)
        if self.num_steps_per_env is None:
            raise KeyError(
                "num_steps_per_env must be defined in the runner or top-level train config"
            )
        if self.save_interval is None:
            raise KeyError(
                "save_interval must be defined in the runner or top-level train config"
            )
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(
                shape=[num_obs], until=1.0e8
            ).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(
                shape=[num_critic_obs], until=1.0e8
            ).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.logger_type = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        self.export_policy_as_onnx = bool(
            self.cfg.get(
                "export_policy_as_onnx",
                train_cfg.get("export_policy_as_onnx", False),
            )
        )
        # Ensure environments are initialised consistently with the base
        # runner implementation.
        if hasattr(self.env, "reset"):
            self.env.reset()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get(
                "logger", self.train_cfg.get("logger", "tensorboard")
            )
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.train_cfg
                )
                self.writer.log_config(
                    self.env.cfg, self.train_cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                import wandb

                # Update the run name with a sequence number. This function is useful to
                # replicate the same behaviour of rsl-rl-lib before v2.3.0
                def update_run_name_with_sequence(prefix: str) -> None:
                    # Retrieve the current wandb run details (project and entity)
                    project = wandb.run.project
                    entity = wandb.run.entity

                    # Use wandb's API to list all runs in your project
                    api = wandb.Api()
                    runs = api.runs(f"{entity}/{project}")

                    max_num = 0
                    # Iterate through runs to extract the numeric suffix after the prefix.
                    for run in runs:
                        if run.name.startswith(prefix):
                            # Extract the numeric part from the run name.
                            numeric_suffix = run.name[
                                len(prefix) :
                            ]  # e.g., from "prefix564", get "564"
                            try:
                                run_num = int(numeric_suffix)
                                if run_num > max_num:
                                    max_num = run_num
                            except ValueError:
                                continue

                    # Increment to get the new run number
                    new_num = max_num + 1
                    new_run_name = f"{prefix}{new_num}"

                    # Update the wandb run's name
                    wandb.run.name = new_run_name
                    print("Updated run name to:", wandb.run.name)

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.train_cfg
                )
                prefix = self.cfg.get(
                    "wandb_project", self.train_cfg.get("wandb_project")
                )
                if prefix is None:
                    raise KeyError(
                        "wandb_project must be specified in the runner or training config when using the wandb logger"
                    )
                update_run_name_with_sequence(prefix=prefix)

                wandb.gym.monitor()
                self.writer.log_config(
                    self.env.cfg, self.train_cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10
                )
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        obs_extras = self._require_observation_extras()
        amp_obs = obs_extras["amp"]
        critic_obs = obs_extras.get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        amp_obs = amp_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout

            mean_style_reward_log = 0.0
            mean_task_reward_log = 0.0
            mean_style_reward_weighted_log = 0.0
            mean_task_reward_weighted_log = 0.0

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    self.alg.act_amp(amp_obs)
                    step_outcome = self.env.step(actions)
                    if len(step_outcome) == 5:
                        obs, critic_source, rewards, dones, infos = step_outcome
                    elif len(step_outcome) == 4:
                        obs, rewards, dones, infos = step_outcome
                        critic_source = None
                    else:
                        raise RuntimeError(
                            "Environment.step must return 4 or 5 values (obs[, critic_obs], reward, done, info)"
                        )

                    next_amp_obs = infos.get("observations", {}).get("amp")
                    if next_amp_obs is None:
                        raise KeyError(
                            "Environment info must contain AMP observations under infos['observations']['amp']"
                        )

                    obs = self.obs_normalizer(obs)
                    if critic_source is None:
                        critic_source = infos.get("observations", {}).get("critic")
                    if critic_source is not None:
                        critic_obs = self.critic_obs_normalizer(critic_source)
                    else:
                        critic_obs = obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    next_amp_obs = next_amp_obs.to(self.device)

                    # Process the AMP reward
                    style_rewards = self.discriminator.predict_reward(
                        amp_obs, next_amp_obs, normalizer=self.amp_normalizer
                    )

                    task_reward_mean = rewards.mean().item()
                    style_reward_mean = style_rewards.mean().item()
                    mean_task_reward_log += task_reward_mean
                    mean_style_reward_log += style_reward_mean

                    weighted_task_mean = self.task_reward_weight * task_reward_mean
                    weighted_style_mean = (
                        self.style_reward_weight * style_reward_mean
                    )
                    mean_task_reward_weighted_log += weighted_task_mean
                    mean_style_reward_weighted_log += weighted_style_mean

                    # Combine the task and style rewards only if style rewards are enabled
                    if self.discriminator.reward_scale != 0:
                        rewards = (
                            self.task_reward_weight * rewards
                            + self.style_reward_weight * style_rewards
                        )

                    self.alg.process_env_step(rewards, dones, infos)
                    self.alg.process_amp_step(next_amp_obs)

                    # The next observation becomes the current observation for the next step
                    amp_obs = torch.clone(next_amp_obs)

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_style_reward_log /= self.num_steps_per_env
            mean_task_reward_log /= self.num_steps_per_env
            mean_style_reward_weighted_log /= self.num_steps_per_env
            mean_task_reward_weighted_log /= self.num_steps_per_env
            mean_total_reward_log = (
                mean_style_reward_weighted_log + mean_task_reward_weighted_log
            )

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_amp_loss,
                mean_grad_pen_loss,
                mean_policy_pred,
                mean_expert_pred,
                mean_accuracy_policy,
                mean_accuracy_expert,
                mean_kl_divergence,
                mean_disc_agent_acc,
                mean_disc_demo_acc,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if self.save_interval and it % self.save_interval == 0:
                self.save(
                    os.path.join(self.log_dir, f"model_{it}.pt"),
                    save_onnx=self.export_policy_as_onnx,
                )
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.current_learning_iteration = tot_iter
        self.save(
            os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"),
            save_onnx=self.export_policy_as_onnx,
        )

    def _require_observation_extras(self, extras: dict | None = None) -> dict:
        """Retrieve the observation extras dictionary with AMP features."""

        if extras is None:
            extras = getattr(self.env, "extras", None)
        if not isinstance(extras, dict):
            raise KeyError("Environment must expose extras dictionary with AMP observations")

        obs_extras = extras.get("observations")
        if not isinstance(obs_extras, dict):
            raise KeyError(
                "Environment extras must contain an 'observations' dictionary for AMP runner"
            )

        if "amp" not in obs_extras:
            raise KeyError(
                "Environment observations must include AMP features under extras['observations']['amp']"
            )

        return obs_extras

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )

        # Adding logging due to AMP
        self.writer.add_scalar("Loss/amp_loss", locs["mean_amp_loss"], locs["it"])
        self.writer.add_scalar(
            "Loss/grad_pen_loss", locs["mean_grad_pen_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/policy_pred", locs["mean_policy_pred"], locs["it"])
        self.writer.add_scalar("Loss/expert_pred", locs["mean_expert_pred"], locs["it"])
        self.writer.add_scalar(
            "Loss/accuracy_policy", locs["mean_accuracy_policy"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/accuracy_expert", locs["mean_accuracy_expert"], locs["it"]
        )
        
        # TokenHSI-style discriminator accuracy logging
        self.writer.add_scalar("Loss/disc_agent_acc", locs["mean_disc_agent_acc"], locs["it"])
        self.writer.add_scalar("Loss/disc_demo_acc", locs["mean_disc_demo_acc"], locs["it"])
        
        # Discriminator separation
        discriminator_separation = locs["mean_expert_pred"] - locs["mean_policy_pred"]
        self.writer.add_scalar("Loss/discriminator_separation", discriminator_separation, locs["it"])

        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar(
            "Loss/mean_kl_divergence", locs["mean_kl_divergence"], locs["it"]
        )
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_style_reward", locs["mean_style_reward_log"], locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_task_reward", locs["mean_task_reward_log"], locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_style_reward_weighted",
                locs["mean_style_reward_weighted_log"],
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_task_reward_weighted",
                locs["mean_task_reward_weighted_log"],
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_total_reward", locs["mean_total_reward_log"], locs["it"]
            )
            if (
                self.logger_type != "wandb"
            ):  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                f"""{'Grad penalty loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                f"""{'Disc agent acc:':>{pad}} {locs['mean_disc_agent_acc']:.4f}\n"""
                f"""{'Disc demo acc:':>{pad}} {locs['mean_disc_demo_acc']:.4f}\n"""
                f"""{'Disc separation:':>{pad}} {discriminator_separation:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""{'Mean task reward (raw):':>{pad}} {locs['mean_task_reward_log']:.2f}\n"""
                f"""{'Mean task reward (weighted):':>{pad}} {locs['mean_task_reward_weighted_log']:.2f}\n"""
                f"""{'Mean style reward (raw):':>{pad}} {locs['mean_style_reward_log']:.2f}\n"""
                f"""{'Mean style reward (weighted):':>{pad}} {locs['mean_style_reward_weighted_log']:.2f}\n"""
                f"""{'Mean total reward:':>{pad}} {locs['mean_total_reward_log']:.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                f"""{'Grad penalty loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                f"""{'Disc agent acc:':>{pad}} {locs['mean_disc_agent_acc']:.4f}\n"""
                f"""{'Disc demo acc:':>{pad}} {locs['mean_disc_demo_acc']:.4f}\n"""
                f"""{'Disc separation:':>{pad}} {discriminator_separation:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string

        # make the eta in H:M:S
        eta_seconds = (
            self.tot_time
            / (locs["it"] + 1)
            * (locs["num_learning_iterations"] - locs["it"])
        )

        # Convert seconds to H:M:S
        eta_h, rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(rem, 60)

        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None, save_onnx=False):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "amp_normalizer": self.alg.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = (
                self.critic_obs_normalizer.state_dict()
            )
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

        if save_onnx:
            # Save the model in ONNX format
            # extract the folder path
            onnx_folder = os.path.dirname(path)

            # extract the iteration number from the path. The path is expected to be in the format
            # model_{iteration}.pt
            iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
            onnx_model_name = f"policy_{iteration}.onnx"

            try:
                export_policy_as_onnx(
                    self.alg.actor_critic,
                    normalizer=self.obs_normalizer,
                    path=onnx_folder,
                    filename=onnx_model_name,
                )
            except ModuleNotFoundError as exc:
                print(
                    f"Warning: skipping ONNX export because a dependency is missing: {exc}"
                )
            except onnx_errors.OnnxExporterError as exc:
                if "Module onnx is not installed" in str(exc):
                    print(
                        "Warning: skipping ONNX export because the onnx package is not installed"
                    )
                else:
                    raise

            if self.logger_type in ["neptune", "wandb"]:
                self.writer.save_model(
                    os.path.join(onnx_folder, onnx_model_name),
                    self.current_learning_iteration,
                )

    def load(self, path, load_optimizer=True, weights_only=False):
        loaded_dict = torch.load(
            path, map_location=self.device, weights_only=weights_only
        )
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(
                loaded_dict["discriminator_state_dict"]
            )
        if "amp_normalizer" in loaded_dict:
            self.alg.amp_normalizer = loaded_dict["amp_normalizer"]

        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(
                loaded_dict["critic_obs_norm_state_dict"]
            )
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            opt_state = loaded_dict["optimizer_state_dict"]
            if len(opt_state.get("param_groups", [])) == len(
                self.alg.optimizer.param_groups
            ):
                try:
                    self.alg.optimizer.load_state_dict(opt_state)
                except ValueError:
                    print("Warning: optimizer state dict incompatible, skipping load")
            else:
                print("Warning: optimizer param group mismatch, skipping optimizer load")
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.empirical_normalization:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(
                self.obs_normalizer(x)
            )  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        self.alg.discriminator.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        self.alg.discriminator.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)