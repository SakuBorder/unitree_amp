# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import List, Union, Tuple, Generator, Optional

import joblib
from dataclasses import dataclass

import torch
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from isaacgym.torch_utils import quat_rotate, quat_mul, quat_from_angle_axis
from phc.phc.utils import torch_utils


@torch.jit.script
def dof_to_obs(pose: torch.Tensor, dof_obs_size: int, dof_offsets: List[int]) -> torch.Tensor:
    """Convert joint angles to the 6D tan-norm representation used by AMP."""
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs = pose.new_zeros(pose.shape[0], dof_obs_size)
    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset : dof_offset + dof_size]

        if dof_size == 3:
            joint_quat = torch_utils.exp_map_to_quat(joint_pose)
        elif dof_size == 1:
            axis = torch.tensor([0.0, 1.0, 0.0], device=pose.device, dtype=pose.dtype)
            joint_quat = quat_from_angle_axis(joint_pose[:, 0], axis)
        else:
            raise RuntimeError("Unsupported joint type")

        joint_obs = torch_utils.quat_to_tan_norm(joint_quat)
        start = j * joint_obs_size
        dof_obs[:, start : start + joint_obs_size] = joint_obs

    return dof_obs


def download_amp_dataset_from_hf(
    destination_dir: Path,
    robot_folder: str,
    files: list,
    repo_id: str = "ami-iit/amp-dataset",
) -> list:
    """
    Downloads AMP dataset files from Hugging Face and saves them to `destination_dir`.
    Ensures real file copies (not symlinks or hard links).

    Args:
        destination_dir (Path): Local directory to save the files.
        robot_folder (str): Folder in the Hugging Face dataset repo to pull from.
        files (list): List of filenames to download.
        repo_id (str): Hugging Face repository ID. Default is "ami-iit/amp-dataset".

    Returns:
        List[str]: List of dataset names (without .npy extension).
    """
    from huggingface_hub import hf_hub_download

    destination_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = []

    for file in files:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{robot_folder}/{file}",
            repo_type="dataset",
            local_files_only=False,
        )
        local_copy = destination_dir / file
        # Deep copy to avoid symlinks
        with open(file_path, "rb") as src_file, open(local_copy, "wb") as dst_file:
            dst_file.write(src_file.read())
        dataset_names.append(file.replace(".npy", ""))

    return dataset_names


@dataclass
class MotionData:
    """
    Data class representing motion data for humanoid agents.

    This class stores joint positions and velocities, base velocities (both in local
    and mixed/world frames), and base orientation (as quaternion). It offers utilities
    for preparing data in AMP-compatible format, as well as environment reset states.

    Attributes:
        - joint_positions: shape (T, N)
        - joint_velocities: shape (T, N)
        - base_lin_velocities_mixed: linear velocity in world frame
        - base_ang_velocities_mixed: (currently zeros)
        - base_lin_velocities_local: linear velocity in local (body) frame
        - base_ang_velocities_local: (currently zeros)
        - base_quat: orientation quaternion as torch.Tensor in xyzw order
        - base_pos: root position in world frame
        - has_foot_positions: whether foot end-effector positions are available

    Notes:
        - The quaternion is expected in the dataset as `xyzw` format (SciPy default),
          and it is converted to a torch tensor in the same ordering.
        - All data is converted to torch.Tensor on the specified device during initialization.
    """

    joint_positions: Union[torch.Tensor, np.ndarray]
    joint_velocities: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_local: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_local: Union[torch.Tensor, np.ndarray]
    base_quat: Union[Rotation, torch.Tensor]
    base_pos: Union[torch.Tensor, np.ndarray]
    foot_positions: Union[torch.Tensor, np.ndarray, None] = None
    device: torch.device = torch.device("cpu")
    has_foot_positions: bool = False
    _warned_missing_feet: bool = False

    def __post_init__(self) -> None:
        # Convert numpy arrays (or SciPy Rotations) to torch tensors
        def to_tensor(x):
            return torch.tensor(x, device=self.device, dtype=torch.float32)

        if isinstance(self.joint_positions, np.ndarray):
            self.joint_positions = to_tensor(self.joint_positions)
        if isinstance(self.joint_velocities, np.ndarray):
            self.joint_velocities = to_tensor(self.joint_velocities)
        if isinstance(self.base_lin_velocities_mixed, np.ndarray):
            self.base_lin_velocities_mixed = to_tensor(self.base_lin_velocities_mixed)
        if isinstance(self.base_ang_velocities_mixed, np.ndarray):
            self.base_ang_velocities_mixed = to_tensor(self.base_ang_velocities_mixed)
        if isinstance(self.base_lin_velocities_local, np.ndarray):
            self.base_lin_velocities_local = to_tensor(self.base_lin_velocities_local)
        if isinstance(self.base_ang_velocities_local, np.ndarray):
            self.base_ang_velocities_local = to_tensor(self.base_ang_velocities_local)
        if isinstance(self.base_quat, np.ndarray):
            self.base_quat = to_tensor(self.base_quat)
        elif isinstance(self.base_quat, Rotation):
            quat_xyzw = self.base_quat.as_quat()  # (T,4) xyzw
            self.base_quat = torch.tensor(
                quat_xyzw,
                device=self.device,
                dtype=torch.float32,
            )
        if isinstance(self.base_pos, np.ndarray):
            self.base_pos = to_tensor(self.base_pos)
        if isinstance(self.foot_positions, np.ndarray):
            self.foot_positions = to_tensor(self.foot_positions)

    def __len__(self) -> int:
        return self.joint_positions.shape[0]

    def get_amp_dataset_obs(self, indices: torch.Tensor) -> torch.Tensor:
        """Return AMP observations for given frames.

        The observation layout mirrors ``TiV2AMP`` and ``HumanoidAMP``: root
        height, root orientation (tan-norm, heading removed), root linear and
        angular velocity expressed in the heading-aligned frame, joint positions
        encoded via ``dof_to_obs`` (6D tan-norm per joint), joint velocities,
        and key body positions (feet) relative to the root in the same local
        frame.  If the dataset lacks foot positions the terms are zero-filled
        and a one-time warning is printed.
        """

        root_pos = self.base_pos[indices]
        root_rot = self.base_quat[indices]
        heading_inv = torch_utils.calc_heading_quat_inv(root_rot)
        root_rot_obs = quat_mul(heading_inv, root_rot)
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

        local_root_vel = quat_rotate(heading_inv, self.base_lin_velocities_mixed[indices])
        local_root_ang_vel = quat_rotate(heading_inv, self.base_ang_velocities_mixed[indices])

        dof_obs_size = self.joint_positions.shape[1] * 6
        dof_offsets = list(range(self.joint_positions.shape[1] + 1))
        dof_obs = dof_to_obs(self.joint_positions[indices], dof_obs_size, dof_offsets)
        dof_vel = self.joint_velocities[indices]

        if self.has_foot_positions:
            feet_world = self.foot_positions[indices].view(indices.shape[0], -1, 3)
            root_pos_exp = root_pos.unsqueeze(1)
            feet_rel = feet_world - root_pos_exp
            heading_exp = heading_inv.unsqueeze(1).repeat(1, feet_rel.shape[1], 1)
            feet_local = quat_rotate(
                heading_exp.view(-1, 4), feet_rel.view(-1, 3)
            ).view(indices.shape[0], -1)
        else:
            if not self._warned_missing_feet:
                print(
                    "[AMPLoader] Motion dataset lacks foot positions; filling with zeros",
                    flush=True,
                )
                self._warned_missing_feet = True
            feet_local = torch.zeros((indices.shape[0], 6), device=self.device)

        obs = [
            root_pos[:, 2:3],
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            feet_local,
        ]

        return torch.cat(obs, dim=1)

    def get_state_for_reset(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns the full state needed for environment reset.

        Args:
            indices: indices of samples to retrieve

        Returns:
            Tuple of (quat, joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities)
        """
        return (
            self.base_quat[indices],
            self.joint_positions[indices],
            self.joint_velocities[indices],
            self.base_lin_velocities_local[indices],
            self.base_ang_velocities_local[indices],
        )

    def get_random_sample_for_reset(self, items: int = 1) -> Tuple[torch.Tensor, ...]:
        indices = torch.randint(0, len(self), (items,), device=self.device)
        return self.get_state_for_reset(indices)


class AMPLoader:
    """
    Loader and processor for humanoid motion capture datasets in AMP format.

    Responsibilities:
      - Loading .npy files containing motion data
      - Building a unified joint ordering across all datasets
      - Resampling trajectories to match the simulator's timestep
      - Computing derived quantities (velocities, local-frame motion)
      - Returning torch-friendly MotionData instances

    Dataset format:
        Each .npy contains a dict with keys:
          - "joints_list": List[str]
          - "joint_positions": List[np.ndarray]
          - "root_position": List[np.ndarray]
          - "root_quaternion": List[np.ndarray] (xyzw)
          - "fps": float (frames/sec)

    Args:
        device: Target torch device ('cpu' or 'cuda')
        dataset_path_root: Directory containing the .npy motion files
        dataset_names: List of dataset filenames (no extension)
        dataset_weights: List of sampling weights (for minibatch sampling)
        simulation_dt: Timestep used by the simulator
        slow_down_factor: Integer factor to slow down original data
        expected_joint_names: (Optional) override for joint ordering
    """

    def __init__(
        self,
        device: str,
        dataset_path_root: Path,
        dataset_names: List[str],
        dataset_weights: List[float],
        simulation_dt: float,
        slow_down_factor: int,
        expected_joint_names: Union[List[str], None] = None,
        expected_foot_body_names: Union[List[str], None] = None,
    ) -> None:
        self.device = device
        if isinstance(dataset_path_root, str):
            dataset_path_root = Path(dataset_path_root)

        # A valid AMP dataset must be provided.  Earlier versions fabricated a
        # dummy all-zero dataset when `dataset_names` was empty, which silently
        # degraded training to pure PPO.  Since we now rely on genuine motion
        # data (e.g. generated via `phc/scripts/data_process/fit_smpl_motion.py`
        # with the `taihu_12dof.yaml` robot config), treat the absence of
        # datasets as an error so the user is forced to supply real files.
        if len(dataset_names) == 0:
            raise ValueError(
                "No AMP datasets provided. Generate motion files with "
                "phc/scripts/data_process/fit_smpl_motion.py and set"
                " `dataset_names` in the training config."
            )
        # ─────────────────────────────────────────────────────────

        # ─── Build union of all joint names if not provided ───
        if expected_joint_names is None:
            joint_union: List[str] = []
            seen = set()
            for name in dataset_names:
                p = dataset_path_root / f"{name}.npy"
                if p.exists():
                    info = np.load(str(p), allow_pickle=True).item()
                    for j in info["joints_list"]:
                        if j not in seen:
                            seen.add(j)
                            joint_union.append(j)
                else:
                    p = dataset_path_root / f"{name}.pkl"
                    if not p.exists():
                        raise FileNotFoundError(p)
                    # pkl files produced by `fit_smpl_motion.py` do not store
                    # joint-name metadata.  In this case the expected list must
                    # be provided explicitly by the caller.
                    raise ValueError(
                        "`expected_joint_names` must be provided when using"
                        " motion datasets saved as .pkl files"
                    )
            expected_joint_names = joint_union
        # ─────────────────────────────────────────────────────────

        # Load and process each dataset into MotionData
        self.expected_foot_body_names = expected_foot_body_names

        self.motion_data: List[MotionData] = []
        for dataset_name in dataset_names:
            dataset_path = dataset_path_root / f"{dataset_name}.npy"
            if not dataset_path.exists():
                dataset_path = dataset_path_root / f"{dataset_name}.pkl"
            if not dataset_path.exists():
                raise FileNotFoundError(dataset_path)
            md = self.load_data(
                dataset_path,
                simulation_dt,
                slow_down_factor,
                expected_joint_names,
                expected_foot_body_names,
            )
            # Provide light‑weight diagnostics so users can verify that the
            # dataset was actually discovered and parsed correctly.  This is
            # particularly helpful when debugging missing or mis-formatted AMP
            # clips.
            sample = md.get_amp_dataset_obs(torch.tensor([0], device=self.device))
            print(
                f"[AMPLoader] Loaded '{dataset_path.name}' with {len(md)} frames "
                f"and {sample.shape[1]} AMP features",
                flush=True,
            )
            self.motion_data.append(md)

        # Normalize dataset-level sampling weights
        weights = torch.tensor(dataset_weights, dtype=torch.float32, device=self.device)
        self.dataset_weights = weights / weights.sum()

        # Precompute flat buffers for fast sampling
        obs_list, next_obs_list, reset_states = [], [], []
        for data, w in zip(self.motion_data, self.dataset_weights):
            T = len(data)
            idx = torch.arange(T, device=self.device)
            obs = data.get_amp_dataset_obs(idx)
            next_idx = torch.clamp(idx + 1, max=T - 1)
            next_obs = data.get_amp_dataset_obs(next_idx)

            obs_list.append(obs)
            next_obs_list.append(next_obs)

            quat, jp, jv, blv, bav = data.get_state_for_reset(idx)
            reset_states.append(torch.cat([quat, jp, jv, blv, bav], dim=1))

        self.all_obs = torch.cat(obs_list, dim=0)
        self.all_next_obs = torch.cat(next_obs_list, dim=0)
        self.all_states = torch.cat(reset_states, dim=0)

        # Build per-frame sampling weights: weight_i / length_i
        lengths = [len(d) for d in self.motion_data]
        per_frame = torch.cat(
            [
                torch.full((L,), w / L, device=self.device)
                for w, L in zip(self.dataset_weights, lengths)
            ]
        )
        self.per_frame_weights = per_frame / per_frame.sum()

    def pad_observations(self, target_dim: int) -> None:
        """Pad the precomputed observation buffers with zeros to match
        ``target_dim``.  This is mainly used when a dataset provides fewer AMP
        features than the environment expects.  The padding keeps the AMP
        pipeline active (e.g. for debugging) while making the mismatch explicit
        in the console output.
        """

        cur_dim = self.all_obs.shape[1]
        if cur_dim == target_dim:
            return
        if cur_dim > target_dim:
            raise ValueError(
                f"AMP dataset has {cur_dim} features but environment expects {target_dim}"
            )

        pad = target_dim - cur_dim

        joint_dim = self.motion_data[0].joint_positions.shape[1]
        foot_dim = (
            3 * len(self.expected_foot_body_names)
            if self.expected_foot_body_names is not None
            else 6
        )

        # Expected layout based on the environment's requirements.  Joint
        # positions are encoded in tan-norm form (6 values per joint) and root
        # orientation uses a 6D representation as well.  Compare these expected
        # sizes against the dataset's provided dimensions to surface what was
        # padded with zeros.
        expected_layout = [
            ("root height", 1, 1),  # (name, expected, provided)
            ("root rotation", 6, 4),
            ("base linear velocity", 3, 3),
            ("base angular velocity", 3, 3),
            ("joint positions", joint_dim * 6, joint_dim),
            ("joint velocities", joint_dim, joint_dim),
            ("foot positions", foot_dim, foot_dim),
        ]

        missing: List[str] = []
        for name, expected, provided in expected_layout:
            if expected > provided:
                missing.append(f"{name} (+{expected - provided})")

        desc = ", ".join(missing)
        print(
            f"[AMPLoader] Padding observations from {cur_dim} to {target_dim} "
            f"features with {pad} zeros ({desc})",
            flush=True,
        )

        self.all_obs = torch.nn.functional.pad(self.all_obs, (0, pad))
        self.all_next_obs = torch.nn.functional.pad(self.all_next_obs, (0, pad))

    def _resample_data_Rn(
        self,
        data: List[np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> np.ndarray:
        f = interp1d(original_keyframes, data, axis=0)
        return f(target_keyframes)

    def _resample_data_SO3(
        self,
        raw_quaternions: List[np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> Rotation:

        # the quaternion is expected in the dataset as `xyzw` format (SciPy default)
        tmp = Rotation.from_quat(raw_quaternions)
        slerp = Slerp(original_keyframes, tmp)
        return slerp(target_keyframes)

    def _compute_ang_vel(
        self,
        data: List[Rotation],
        dt: float,
        local: bool = False,
    ) -> np.ndarray:
        R_prev = data[:-1]
        R_next = data[1:]

        if local:
            # Exp = R_i⁻¹ · R_{i+1}
            rel = R_prev.inv() * R_next
        else:
            # Exp = R_{i+1} · R_i⁻¹
            rel = R_next * R_prev.inv()

        # Log-map to rotation vectors and divide by Δt
        rotvec = rel.as_rotvec() / dt

        return np.vstack((rotvec, rotvec[-1]))

    def _compute_raw_derivative(self, data: np.ndarray, dt: float) -> np.ndarray:
        d = (data[1:] - data[:-1]) / dt
        return np.vstack([d, d[-1:]])

    def load_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int = 1,
        expected_joint_names: Union[List[str], None] = None,
        expected_foot_body_names: Union[List[str], None] = None,
    ) -> MotionData:
        """
        Loads and processes one motion dataset.

        Returns:
            MotionData instance
        """
        if dataset_path.suffix == ".pkl":
            raw = joblib.load(dataset_path)
            if isinstance(raw, dict) and len(raw) == 1:
                data = next(iter(raw.values()))
            else:
                data = raw

            dataset_joint_names = expected_joint_names or []
            jp_list = [frame for frame in data["dof"]]
            root_pos = data["root_trans_offset"]
            root_quat = data["root_rot"]
            fps = data["fps"]
            foot_pos = data.get("foot_positions")
            foot_body_names = data.get("foot_body_names")
            if foot_pos is not None and foot_body_names is None:
                foot_body_names = data.get("dof_names") or dataset_joint_names
                print(
                    "[AMPLoader] Foot positions provided without names; "
                    "using joint names as candidates",
                    flush=True,
                )
        else:
            data = np.load(str(dataset_path), allow_pickle=True).item()
            dataset_joint_names = data["joints_list"]

            # build index map for expected_joint_names
            idx_map: List[Union[int, None]] = []
            for j in expected_joint_names:
                if j in dataset_joint_names:
                    idx_map.append(dataset_joint_names.index(j))
                else:
                    idx_map.append(None)

            # reorder & fill joint positions
            jp_list: List[np.ndarray] = []
            for frame in data["joint_positions"]:
                arr = np.zeros((len(idx_map),), dtype=frame.dtype)
                for i, src_idx in enumerate(idx_map):
                    if src_idx is not None:
                        arr[i] = frame[src_idx]
                jp_list.append(arr)

            root_pos = data["root_position"]
            root_quat = data["root_quaternion"]
            fps = data["fps"]
            foot_pos = data.get("foot_positions")
            foot_body_names = data.get("foot_body_names")
            if foot_pos is not None and foot_body_names is None:
                foot_body_names = data.get("dof_names") or dataset_joint_names
                print(
                    "[AMPLoader] Foot positions provided without names; "
                    "using joint names as candidates",
                    flush=True,
                )

        dt = 1.0 / fps / float(slow_down_factor)
        T = len(jp_list)
        t_orig = np.linspace(0, T * dt, T)
        T_new = int(T * dt / simulation_dt)
        t_new = np.linspace(0, T * dt, T_new)

        resampled_joint_positions = self._resample_data_Rn(jp_list, t_orig, t_new)
        resampled_joint_velocities = self._compute_raw_derivative(
            resampled_joint_positions, simulation_dt
        )

        resampled_base_positions = self._resample_data_Rn(root_pos, t_orig, t_new)
        resampled_base_orientations = self._resample_data_SO3(
            root_quat, t_orig, t_new
        )

        resampled_base_lin_vel_mixed = self._compute_raw_derivative(
            resampled_base_positions, simulation_dt
        )

        resampled_base_ang_vel_mixed = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=False
        )

        resampled_base_lin_vel_local = np.stack(
            [
                R.as_matrix().T @ v
                for R, v in zip(
                    resampled_base_orientations, resampled_base_lin_vel_mixed
                )
            ]
        )
        resampled_base_ang_vel_local = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=True
        )

        foot_positions_flat = None
        has_feet = False
        if foot_pos is not None and expected_foot_body_names is not None and foot_body_names is not None:
            idx_map: List[Union[int, None]] = []
            for n in expected_foot_body_names:
                if n in foot_body_names:
                    src_idx = foot_body_names.index(n)
                    idx_map.append(src_idx)
                    sample = foot_pos[0][src_idx].tolist()
                    print(
                        f"[AMPLoader] Foot body '{n}' -> idx {src_idx}, sample {sample}",
                        flush=True,
                    )
                else:
                    idx_map.append(None)
                    print(
                        f"[AMPLoader] Foot body '{n}' not found in dataset", flush=True
                    )
            feet_reordered: List[np.ndarray] = []
            for frame in foot_pos:
                arr = np.zeros((len(expected_foot_body_names), 3), dtype=frame.dtype)
                for i, src in enumerate(idx_map):
                    if src is not None:
                        arr[i] = frame[src]
                feet_reordered.append(arr.reshape(-1))
            resampled_feet = self._resample_data_Rn(feet_reordered, t_orig, t_new)
            foot_positions_flat = resampled_feet
            has_feet = True
        elif expected_foot_body_names is not None:
            candidates = foot_body_names if foot_body_names is not None else dataset_joint_names
            if candidates:
                print(
                    f"[AMPLoader] Available foot candidates: {list(candidates)}",
                    flush=True,
                )
            for n in expected_foot_body_names:
                print(
                    f"[AMPLoader] Foot body '{n}' not provided in dataset", flush=True
                )

        return MotionData(
            joint_positions=resampled_joint_positions,
            joint_velocities=resampled_joint_velocities,
            base_lin_velocities_mixed=resampled_base_lin_vel_mixed,
            base_ang_velocities_mixed=resampled_base_ang_vel_mixed,
            base_lin_velocities_local=resampled_base_lin_vel_local,
            base_ang_velocities_local=resampled_base_ang_vel_local,
            base_quat=resampled_base_orientations,
            base_pos=resampled_base_positions,
            foot_positions=foot_positions_flat,
            device=self.device,
            has_foot_positions=has_feet,
        )

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Yields mini-batches of (state, next_state) pairs for training,
        sampled directly from precomputed buffers.

        Args:
            num_mini_batch: Number of mini-batches to yield
            mini_batch_size: Size of each mini-batch
        Yields:
            Tuple of (state, next_state) tensors
        """
        for _ in range(num_mini_batch):
            idx = torch.multinomial(
                self.per_frame_weights, mini_batch_size, replacement=True
            )
            yield self.all_obs[idx], self.all_next_obs[idx]

    def get_state_for_reset(self, number_of_samples: int) -> Tuple[torch.Tensor, ...]:
        """
        Randomly samples full states for environment resets,
        sampled directly from the precomputed state buffer.

        Args:
            number_of_samples: Number of samples to retrieve
        Returns:
            Tuple of (quat, joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities)
        """
        idx = torch.multinomial(
            self.per_frame_weights, number_of_samples, replacement=True
        )
        full = self.all_states[idx]
        joint_dim = self.motion_data[0].joint_positions.shape[1]

        # The dimensions of the full state are:
        #   - 4 (quat) + joint_dim (joint_positions) + joint_dim (joint_velocities)
        #   + 3 (base_lin_velocities) + 3 (base_ang_velocities)
        #   = 4 + joint_dim + joint_dim + 3 + 3
        dims = [4, joint_dim, joint_dim, 3, 3]
        return torch.split(full, dims, dim=1)