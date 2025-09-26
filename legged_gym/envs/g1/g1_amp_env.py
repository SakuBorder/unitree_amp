"""AMP-enabled environment for the Unitree G1 humanoid."""

import fnmatch
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

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

        self._refresh_observation_extras()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self._amp_obs_buf is not None and len(env_ids) > 0:
            self._amp_obs_buf[env_ids] = 0.0

    def get_observations(self) -> torch.Tensor:
        self._refresh_observation_extras()
        return self.obs_buf

    # ------------------------------------------------------------------
    # AMP utilities
    # ------------------------------------------------------------------
    def fetch_amp_obs_demo(self, num_samples: int, delta_t: float):
        """Sample expert AMP transitions directly from the motion library."""

        if self._motion_lib is None:
            raise RuntimeError("Motion library not initialised; set cfg.motion_lib.motion_file")

        num_steps = int(self.cfg.amp.num_obs_steps)
        if num_steps <= 0:
            raise ValueError("cfg.amp.num_obs_steps must be a positive integer")
        hist_dt = self.dt * max(num_steps - 1, 0)
        truncate_time = delta_t + hist_dt

        motion_ids = self._motion_lib.sample_motions(num_samples)
        base_times = self._motion_lib.sample_time(
            motion_ids, truncate_time=truncate_time if truncate_time > 0.0 else None
        )

        offsets = torch.arange(
            num_steps,
            device=base_times.device,
            dtype=base_times.dtype,
        )
        offsets = hist_dt - offsets * self.dt

        window_ids = motion_ids.unsqueeze(1).expand(-1, num_steps).reshape(-1)
        window_times0 = (base_times.unsqueeze(1) + offsets).reshape(-1)
        window_times1 = window_times0 + delta_t

        state0 = self._motion_lib.get_motion_state(window_ids, window_times0)
        state1 = self._motion_lib.get_motion_state(window_ids, window_times1)

        obs0 = self._build_amp_from_motion_state(state0).view(num_samples, num_steps, -1)
        obs1 = self._build_amp_from_motion_state(state1).view(num_samples, num_steps, -1)

        return obs0.view(num_samples, -1), obs1.view(num_samples, -1)

    # ------------------------------------------------------------------
    # Motion sampling utilities
    # ------------------------------------------------------------------
    def _ensure_iterable(self, value) -> Sequence:
        if value is None:
            return tuple()
        if isinstance(value, (str, bytes)):
            return (value,)
        return tuple(value)

    def _to_group_config(self, group) -> Mapping[str, object]:
        if isinstance(group, Mapping):
            return group
        attrs = {}
        for name in dir(group):
            if name.startswith("_"):
                continue
            value = getattr(group, name)
            if callable(value):
                continue
            attrs[name] = value
        return attrs

    def _build_motion_group_mask(
        self,
        keys: Sequence[str],
        keys_lower: Sequence[str],
        group_cfg: Mapping[str, object],
    ) -> torch.Tensor:
        device = self._motion_lib._device
        mask = [False] * len(keys)

        indices = group_cfg.get("indices")
        if indices is not None:
            for idx in self._ensure_iterable(indices):
                try:
                    mask[int(idx)] = True
                except (TypeError, ValueError, IndexError):
                    continue

        patterns = group_cfg.get("patterns")
        if patterns is None:
            pattern = group_cfg.get("pattern")
            if pattern is not None:
                patterns = (pattern,)
        for pattern in self._ensure_iterable(patterns):
            pat = str(pattern)
            pat_lower = pat.lower()
            for idx, key_lower in enumerate(keys_lower):
                if fnmatch.fnmatch(key_lower, pat_lower):
                    mask[idx] = True

        substrings = group_cfg.get("contains")
        if substrings is None:
            substrings = group_cfg.get("substrings")
        for sub in self._ensure_iterable(substrings):
            sub_lower = str(sub).lower()
            if not sub_lower:
                continue
            for idx, key_lower in enumerate(keys_lower):
                if sub_lower in key_lower:
                    mask[idx] = True

        return torch.tensor(mask, dtype=torch.bool, device=device)

    def _apply_motion_sampling_groups(
        self,
        sampling_groups: Iterable[Mapping[str, object]],
        default_weight: Optional[float],
    ) -> Optional[Dict[str, object]]:
        if not sampling_groups:
            return None
        if self._motion_lib is None:
            return None

        try:
            raw_keys = list(self._motion_lib._motion_data_keys)
        except AttributeError:
            return None

        clip_names = [str(key) for key in raw_keys]
        if not clip_names:
            return None

        clip_names_lower = [name.lower() for name in clip_names]
        device = self._motion_lib._device
        weights = torch.zeros(len(clip_names), device=device, dtype=torch.float32)
        matched = torch.zeros(len(clip_names), device=device, dtype=torch.bool)
        summaries = []

        for group in sampling_groups:
            group_cfg = self._to_group_config(group)
            weight = float(group_cfg.get("weight", 0.0))
            if weight <= 0.0:
                continue

            mask = self._build_motion_group_mask(clip_names, clip_names_lower, group_cfg)
            match_count = int(mask.sum().item())
            group_name = group_cfg.get("name", "")
            if match_count == 0:
                summaries.append({"name": group_name, "matches": 0, "weight": weight})
                continue

            weights[mask] += weight / match_count
            matched |= mask
            summaries.append({"name": group_name, "matches": match_count, "weight": weight})

        unmatched = (~matched).nonzero(as_tuple=False).flatten()
        if unmatched.numel() > 0:
            if default_weight is not None and default_weight > 0.0:
                weights[unmatched] += default_weight / unmatched.numel()
                summaries.append(
                    {
                        "name": "__default__",
                        "matches": int(unmatched.numel()),
                        "weight": float(default_weight),
                    }
                )
            elif weights.sum() <= 0.0:
                weights[:] = 1.0 / len(clip_names)
            else:
                # Allow unmatched clips but keep them with a very small mass so
                # normalisation succeeds.  The scale does not matter because
                # the distribution is renormalised below.
                weights[unmatched] = weights[matched].mean() if matched.any() else 1.0

        total = weights.sum()
        if total <= 0.0:
            weights[:] = 1.0 / len(clip_names)
            total = 1.0

        weights /= total
        self._motion_lib._sampling_prob = weights

        return {
            "clip_names": clip_names,
            "weights": weights.detach().cpu(),
            "groups": summaries,
        }

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

        sampling_groups = getattr(motion_cfg, "sampling_groups", ())
        default_weight = getattr(motion_cfg, "default_sampling_weight", None)
        sampling_summary = self._apply_motion_sampling_groups(sampling_groups, default_weight)

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

        if sampling_summary is not None:
            motion_extras = self.extras.setdefault("motion", {})
            motion_extras["clip_names"] = sampling_summary["clip_names"]
            motion_extras["sampling_weights"] = sampling_summary["weights"]
            motion_extras["sampling_groups"] = sampling_summary["groups"]

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

        key_body_pos = state.get("key_pos")
        if key_body_pos is not None:
            # Some motion libraries provide dedicated key positions that may
            # already match the configured body order. If they instead expose
            # the full rigid-body array, align them with the configured AMP
            # indices for backwards compatibility.
            if key_body_pos.shape[1] != len(self.cfg.amp.key_body_names):
                if self._motion_key_body_indices is None:
                    raise RuntimeError(
                        "Motion key body indices are undefined; cannot align key_pos entries."
                    )
                if (
                    key_body_pos.shape[1]
                    <= int(self._motion_key_body_indices.max().item())
                ):
                    raise ValueError(
                        "Motion key_pos does not cover the configured AMP key bodies."
                    )
                key_body_pos = key_body_pos[:, self._motion_key_body_indices, :]
        if key_body_pos is not None:
            key_body_pos = torch.as_tensor(key_body_pos, device=self.device)
        else:
            if self._motion_key_body_indices is None:
                raise RuntimeError(
                    "Motion key body indices are undefined; check motion library initialisation."
                )
            key_body_pos = state["rg_pos"][:, self._motion_key_body_indices, :]
        key_body_pos = key_body_pos.to(self.device)
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

    def _refresh_observation_extras(self) -> None:
        """Ensure the latest AMP and critic observations are exposed via extras."""

        amp_flat = (
            self._amp_obs_buf.view(self.num_envs, -1)
            if self._amp_obs_buf is not None
            else torch.zeros(self.num_envs, 0, device=self.device)
        )
        obs_extras = self.extras.setdefault("observations", {})
        obs_extras["amp"] = amp_flat
        if self.privileged_obs_buf is not None:
            obs_extras["critic"] = self.privileged_obs_buf