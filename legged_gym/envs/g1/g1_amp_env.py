"""AMP-enabled environment for the Unitree G1 humanoid."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.g1.g1_env import G1Robot

from phc.phc.env.tasks.humanoid_amp import build_amp_observations
from phc.phc.utils.motion_lib_g1 import MotionLibG1, FixHeightMode
from phc.smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree


class G1AMPRobot(G1Robot):
    """Extends the locomotion task with Adversarial Motion Priors (AMP).

    The class maintains a motion library to sample expert demonstrations and
    exposes AMP observations alongside the standard proprioceptive state.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._motion_lib = None
        self._amp_obs_buf = None
        self._amp_key_body_indices = None
        self._motion_key_body_indices = None
        self._amp_obs_per_step = 0
        self._dof_offsets = None

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self._init_motion_lib()
        self._configure_amp_buffers()

    # ---------------------------------------------------------------------
    # Isaac Gym overrides
    # ------------------------------------------------------------------
    def _init_buffers(self):
        super()._init_buffers()
        # Ensure AMP buffers exist even before the first physics step.
        if self._amp_obs_per_step == 0:
            self._dof_offsets = list(range(self.num_dof + 1))
            self._amp_obs_per_step = self._compute_amp_obs_per_step(len(self.cfg.amp.key_body_names))
        self._amp_obs_buf = torch.zeros(
            self.num_envs,
            self.cfg.amp.num_obs_steps,
            self._amp_obs_per_step,
            device=self.device,
            dtype=torch.float,
        )

    def compute_observations(self):
        super().compute_observations()
        if self._amp_obs_buf is None or self._amp_key_body_indices is None:
            return

        curr_obs = self._compute_amp_observation_step()
        self._amp_obs_buf[:, 1:] = self._amp_obs_buf[:, :-1]
        self._amp_obs_buf[:, 0] = curr_obs

        amp_flat = self._amp_obs_buf.view(self.num_envs, -1)
        obs_extras = self.extras.setdefault("observations", {})
        obs_extras["amp"] = amp_flat
        if self.privileged_obs_buf is not None:
            obs_extras["critic"] = self.privileged_obs_buf

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self._amp_obs_buf is not None and len(env_ids) > 0:
            self._amp_obs_buf[env_ids] = 0.0

    def get_observations(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        amp_flat = (
            self._amp_obs_buf.view(self.num_envs, -1)
            if self._amp_obs_buf is not None
            else torch.zeros(self.num_envs, 0, device=self.device)
        )
        extras: Dict[str, torch.Tensor] = {"observations": {"amp": amp_flat}}
        if self.privileged_obs_buf is not None:
            extras["observations"]["critic"] = self.privileged_obs_buf
        return self.obs_buf, extras

    # ------------------------------------------------------------------
    # AMP utilities
    # ------------------------------------------------------------------
    def fetch_amp_obs_demo(self, num_samples: int, delta_t: float):
        """Sample expert AMP transitions directly from the motion library."""

        if self._motion_lib is None:
            raise RuntimeError("Motion library not initialised; set cfg.motion_lib.motion_file")

        motion_ids = self._motion_lib.sample_motions(num_samples)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=delta_t)
        motion_times1 = motion_times0 + delta_t

        state0 = self._motion_lib.get_motion_state(motion_ids, motion_times0)
        state1 = self._motion_lib.get_motion_state(motion_ids, motion_times1)

        obs0 = self._build_amp_from_motion_state(state0)
        obs1 = self._build_amp_from_motion_state(state1)

        return obs0, obs1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_motion_lib(self):
        motion_cfg = self.cfg.motion_lib
        motion_file = motion_cfg.motion_file
        if motion_file is None:
            # Allow the environment to exist without AMP data so that unit tests
            # or lightweight simulations can still run.
            print("[G1AMP] No motion library specified; AMP demos disabled.")
            return

        motion_path = Path(
            motion_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        ).expanduser()

        if motion_path.is_dir():
            if not any(motion_path.glob("*.pkl")):
                raise FileNotFoundError(
                    f"No motion library .pkl files found in directory: {motion_path}."
                )
        elif not motion_path.is_file():
            raise FileNotFoundError(
                f"Motion library file not found: {motion_path}. Update cfg.motion_lib.motion_file."
            )

        mjcf_path = Path(
            motion_cfg.mjcf_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        ).expanduser()
        if not mjcf_path.is_file():
            raise FileNotFoundError(f"MJCF file not found for motion library: {mjcf_path}")

        device = torch.device(self.device)
        fix_height = getattr(FixHeightMode, motion_cfg.fix_height_mode)
        self._motion_lib = MotionLibG1(
            motion_file=str(motion_path),
            device=device,
            fix_height=fix_height,
            multi_thread=motion_cfg.multi_thread,
            mjcf_file=str(mjcf_path),
            sim_timestep=self.dt,
        )

        # Build skeletons for motion loading.
        skeleton = SkeletonTree.from_mjcf(str(mjcf_path))
        num_motions = min(motion_cfg.num_motions_per_batch, self.num_envs)
        skeletons = [skeleton] * num_motions
        gender_betas = [torch.zeros(17)] * num_motions
        limb_weights = [np.zeros(10)] * num_motions
        self._motion_lib.load_motions(
            skeleton_trees=skeletons,
            gender_betas=gender_betas,
            limb_weights=limb_weights,
            random_sample=True,
        )

        self._register_key_body_indices()

    def _register_key_body_indices(self):
        self._ensure_sim_key_body_indices()

        motion_dict = {name: idx for idx, name in enumerate(self._motion_lib.mesh_parsers.model_names)}
        motion_indices = []
        for name in self.cfg.amp.key_body_names:
            if name not in motion_dict:
                raise KeyError(f"Body '{name}' not present in motion library")
            motion_indices.append(motion_dict[name])
        self._motion_key_body_indices = torch.as_tensor(
            motion_indices, dtype=torch.long, device=self._motion_lib._device
        )

        self._dof_offsets = list(range(self.num_dof + 1))
        self._amp_obs_per_step = self._compute_amp_obs_per_step(len(self.cfg.amp.key_body_names))

    def _compute_amp_observation_step(self):
        root_pos = self.root_states[:, 0:3]
        root_rot = self.root_states[:, 3:7]
        root_vel = self.root_states[:, 7:10]
        root_ang_vel = self.root_states[:, 10:13]
        dof_pos = self.dof_pos
        dof_vel = self.dof_vel
        key_body_pos = self.rigid_body_states_view[:, self._amp_key_body_indices, :3]

        return build_amp_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_body_pos,
            self.cfg.amp.local_root_obs,
            self.cfg.amp.root_height_obs,
            self._dof_obs_size,
            self._dof_offsets,
        )

    def _build_amp_from_motion_state(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._motion_key_body_indices is None:
            raise RuntimeError("Motion key body indices are undefined; check motion library initialisation.")

        key_body_pos = state["rg_pos"][:, self._motion_key_body_indices, :]
        obs = build_amp_observations(
            state["root_pos"],
            state["root_rot"],
            state["root_vel"],
            state["root_ang_vel"],
            state["dof_pos"],
            state["dof_vel"],
            key_body_pos,
            self.cfg.amp.local_root_obs,
            self.cfg.amp.root_height_obs,
            self._dof_obs_size,
            self._dof_offsets,
        )
        return obs.to(self.device).view(state["root_pos"].shape[0], -1)

    def _configure_amp_buffers(self):
        self._ensure_sim_key_body_indices()

        if self._amp_obs_per_step == 0:
            num_keys = len(self.cfg.amp.key_body_names)
            self._dof_offsets = list(range(self.num_dof + 1))
            self._amp_obs_per_step = self._compute_amp_obs_per_step(num_keys)
        self._dof_obs_size = (len(self._dof_offsets) - 1) * 6
        if self._amp_obs_buf is None:
            self._amp_obs_buf = torch.zeros(
                self.num_envs,
                self.cfg.amp.num_obs_steps,
                self._amp_obs_per_step,
                device=self.device,
                dtype=torch.float,
            )

    def _compute_amp_obs_per_step(self, num_key_bodies: int) -> int:
        # AMP features: height (1) + root rotation tan-norm (6) + linear vel (3)
        # + angular vel (3) + dof tan-norm (6 per joint) + dof velocity (num_dof)
        # + key body positions (3 per body).
        dof_obs = (self.num_dof) * 6
        return 1 + 6 + 3 + 3 + dof_obs + self.num_dof + 3 * num_key_bodies

    def _ensure_sim_key_body_indices(self):
        if self._amp_key_body_indices is not None:
            return
        body_dict = self.gym.get_actor_rigid_body_dict(self.envs[0], self.actor_handles[0])
        sim_indices = []
        for name in self.cfg.amp.key_body_names:
            if name not in body_dict:
                raise KeyError(f"Body '{name}' not found in simulation asset")
            sim_indices.append(body_dict[name])
        self._amp_key_body_indices = torch.as_tensor(sim_indices, dtype=torch.long, device=self.device)
