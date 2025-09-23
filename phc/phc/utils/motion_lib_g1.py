

import numpy as np
import os
import yaml
from tqdm import tqdm
import os.path as osp

from phc.phc.utils import torch_utils
import joblib
import torch
import torch.multiprocessing as mp
import copy
import gc
from phc.phc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
from scipy.spatial.transform import Rotation as sRot
import random
from phc.phc.utils.flags import flags
from phc.phc.utils.motion_lib_base import MotionLibBase, DeviceCache, compute_motion_dof_vels, FixHeightMode
from phc.phc.utils.torch_g1_humanoid_batch import Humanoid_Batch
from easydict import EasyDict

NUM_ENVS = 8192

def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)



USE_CACHE = True
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    
    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


class MotionLibG1(MotionLibBase):

    def __init__(self, motion_file, device, fix_height=FixHeightMode.no_fix, masterfoot_conifg=None, min_length=-1, im_eval=False, multi_thread=False, extend_hand = False, extend_head = False, mjcf_file="resources/robots/g1/g1_29dof.xml", sim_timestep = 1/50):
        super().__init__(motion_file=motion_file, device=device, fix_height=fix_height, masterfoot_conifg=masterfoot_conifg, min_length=min_length, im_eval=im_eval, multi_thread=multi_thread, sim_timestep = sim_timestep)
        self.mesh_parsers = Humanoid_Batch(extend_hand = extend_hand, extend_head = extend_head, mjcf_file=mjcf_file)
        return
    
    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        
        with torch.no_grad():
            raise NotImplementedError("Fix height is not implemented for G1")
            return trans, diff_fix
        
    def load_motions(self, skeleton_trees, gender_betas, limb_weights, random_sample=True, start_idx=0, max_len=-1, target_heading = None):
        # load motion load the same number of motions as there are skeletons (humanoids)
        # if "gts" in self.__dict__:
        #     del self.gts, self.grs, self.lrs, self.grvs, self.gravs, self.gavs, self.gvs, self.dvs
        #     del self._motion_lengths, self._motion_fps, self._motion_dt, self._motion_num_frames, self._motion_bodies, self._motion_aa
        #     if "gts_t" in self.__dict__:
        #         self.gts_t, self.grs_t, self.gvs_t
        #     if flags.real_traj:
        #         del self.q_gts, self.q_grs, self.q_gavs, self.q_gvs

        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        
        if flags.real_traj:
            self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

        torch.cuda.empty_cache()
        gc.collect()

        total_len = 0.0
        self.num_joints = len(skeleton_trees[0].node_names)
        num_motion_to_load = len(skeleton_trees)
        print("self.num_joints:",self.num_joints)
        print("num_motion_to_load:",num_motion_to_load)

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
        else:
            sample_idxes = torch.remainder(torch.arange(len(skeleton_trees)) + start_idx, self._num_unique_motions ).to(self._device)

        # import ipdb; ipdb.set_trace()
        self._curr_motion_ids = sample_idxes
        self.one_hot_motions = torch.nn.functional.one_hot(self._curr_motion_ids, num_classes = self._num_unique_motions).to(self._device)  # Testing for obs_v5
        self.curr_motion_keys = self._motion_data_keys[sample_idxes]
        # self.curr_motion_keys = self._motion_data_keys[10]
        self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

        print("\n****************************** Current motion keys ******************************")
        # print("Sampling motion:", sample_idxes[:30])
        print("Sampling motion:", sample_idxes)
        if len(self.curr_motion_keys) < 100:
            print(self.curr_motion_keys)
        else:
            print(self.curr_motion_keys[:30], ".....")
        print("*********************************************************************************\n")

        print("_motion_data_keys.size():",len(self._motion_data_keys))
        # print("_motion_data_keys[3932]:",self._motion_data_keys[3932])
        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        # print("motion_data_list[4087]:",motion_data_list[4087])   #TODO:there are something wrong with  'motion_data_list'
        # print("motion_data_list[4088]:",motion_data_list[4088])   #TODO:there are something wrong with  'motion_data_list'
        # print("motion_data_list[3932]:",motion_data_list[3932])   #TODO:there are something wrong with  'motion_data_list'
        # print("motion_data_list[3933]:",motion_data_list[3933])   #TODO:there are something wrong with  'motion_data_list'
        
        #This is a test for memory leak
        # with mp.Manager() as manager:
        mp.set_sharing_strategy('file_descriptor')

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = min(mp.cpu_count(), 64)
        print("num_jobs:",num_jobs) # 64 / 40
        
        print("flags.debug:",flags.debug) #False

        if num_jobs <= 8 or not self.multi_thread: # self.multi_thread = False
            num_jobs = 1
        if flags.debug:
            num_jobs = 1
        
        print("num_jobs:",num_jobs) # 1
        res_acc = {}  # using dictionary ensures order of the results.
        jobs = motion_data_list
        # print("self._motion_data_keys[0]:",self._motion_data_keys[0]) #0-BMLhandball_S01_Expert_Trial_upper_left_right_176_poses
        # print("self._motion_data_keys[3932]:",self._motion_data_keys[3932]) #0-BioMotionLab_NTroje_rub023_0016_sitting2_poses
        # print("self._motion_data_keys[0]:",self._motion_data_keys[0])
        # print("self._motion_data_keys[1]:",self._motion_data_keys[1])
        print("len(jobs):",len(jobs)) # 4096
        chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        print("chunk:",chunk) # 4096
        ids = np.arange(len(jobs))
        # print("jobs[4088]:",jobs[4088]) #-0.00529744
        # print("jobs[3932]:",jobs[3932])
        # print("jobs[3931]:",jobs[3931])
        # print("jobs[0]:",jobs[0])
        # print("jobs[1]:",jobs[1])
        
        
        #jobs = [(ids[0:4096], jobs[0:4096], skeleton_trees[0:4096], gender_betas[0:4096], self.fix_height, self.mesh_parsers, self._masterfoot_conifg, target_heading, max_len)]
        jobs = [(ids[i:i + chunk], jobs[i:i + chunk], skeleton_trees[i:i + chunk], gender_betas[i:i + chunk], self.fix_height, self.mesh_parsers, self._masterfoot_conifg, target_heading, max_len) for i in range(0, len(jobs), chunk)]
        job_args = [jobs[i] for i in range(len(jobs))]
        for i in range(1, len(jobs)):#not work
            worker_args = (*job_args[i], queue, i)
            worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
            worker.start()
            print("test multi process")
        # print("jobs[0]:",jobs[0])
        res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 10))
        
        for i in tqdm(range(len(jobs) - 1)):
            res = queue.get()
            res_acc.update(res)
            
        # print("res_acc[3379].dof_vels.shape:",res_acc[3379].dof_vels.shape)
        # print("res_acc[3379]:",res_acc[3378])

        for f in tqdm(range(len(res_acc))):
            motion_file_data, curr_motion = res_acc[f]
            # print("curr_motion.dof_vels.dim():",curr_motion.dof_vels.dim())
            i = 1
            while(curr_motion.dof_vels.dim() < 2):
                print("$$$$$$$$$$$$$$$$$$$$$$$$\nWARNING:DUE TO LENGTH OF dof_vels IS LESS THAN 2, RESAMPLEING!!!!!!\n$$$$$$$$$$$$$$$$$$$$$$$$")
                print("f:",f)
                print("self._motion_data_keys[f]:",self._motion_data_keys[f])
                if(f+i>=NUM_ENVS):
                    i=-1
                motion_file_data, curr_motion = res_acc[f+i]
                i = i + 1 if i >0 else i - 1
            
            
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)
            
            
            if "beta" in motion_file_data:
                _motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
                _motion_bodies.append(curr_motion.gender_beta)
            else:
                _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                _motion_bodies.append(torch.zeros(17))

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            
            if flags.real_traj:
                self.q_gts.append(curr_motion.quest_motion['quest_trans'])
                self.q_grs.append(curr_motion.quest_motion['quest_rot'])
                self.q_gavs.append(curr_motion.quest_motion['global_angular_vel'])
                self.q_gvs.append(curr_motion.quest_motion['linear_vel'])
                
            del curr_motion
            
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(_motion_aa), device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
        self._motion_limb_weights = torch.tensor(np.array(limb_weights), device=self._device, dtype=torch.float32)
        self._num_motions = len(motions)

        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        # i = 0
        # dvs_list = []
        # for i, m in enumerate(motions):
        #     if m.dof_vels.dim() < 2 :
        #         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\nWARNING: THE MOTION>DOF_VELS IS LESSER THAN 2!!!!!!!!!\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        #         print("_motion_data_keys[]:",i,self._motion_data_keys[i])
        #         print("m:",m)
        #         m.dof_vels = m.dof_vels.unsqueeze(0)
        #         print(f"Index: {i}, Shape: {m.dof_vels.shape}")
        #         print(f"Index: {i}, dim(): {m.dof_vels.dim()}")
        #         input("Press Enter to continue...")
        #     dvs_list.append(m.dof_vels)

        # print("motions[3379]:",motions[3380])
        # self.dvs = torch.cat(dvs_list,dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device) # TODO:crashed here
        
        
        
        
        ############################## below is the optimized code ###############################
        # while(m.dof_vels.dim() < 2):
            
        
        
        
        ############################## above is the optimized code ###############################
        
        if "global_translation_extend" in motions[0].__dict__:
            self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)
            self.grs_t = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)
            self.gvs_t = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)
            self.gavs_t = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)
        
        if "dof_pos" in motions[0].__dict__:
            self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)
        
        if flags.real_traj:
            self.q_gts = torch.cat(self.q_gts, dim=0).float().to(self._device)
            self.q_grs = torch.cat(self.q_grs, dim=0).float().to(self._device)
            self.q_gavs = torch.cat(self.q_gavs, dim=0).float().to(self._device)
            self.q_gvs = torch.cat(self.q_gvs, dim=0).float().to(self._device)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        motion = motions[0]
        self.num_bodies = self.num_joints

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        return motions

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        if "dof_pos" in self.__dict__:
            local_rot0 = self.dof_pos[f0l]
            local_rot1 = self.dof_pos[f1l]
        else:
            local_rot0 = self.lrs[f0l]
            local_rot1 = self.lrs[f1l]
            
        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        

        if "dof_pos" in self.__dict__: # H1 joints
            dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
            dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1
        else:
            dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
            local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        return_dict = {}

        if "gts_t" in self.__dict__: #global trans extend
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]
            
            rg_rot_t0 = self.grs_t[f0l]
            rg_rot_t1 = self.grs_t[f1l]
            
            body_vel_t0 = self.gvs_t[f0l]
            body_vel_t1 = self.gvs_t[f1l]
            
            body_ang_vel_t0 = self.gavs_t[f0l]
            body_ang_vel_t1 = self.gavs_t[f1l]
            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1  
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            rg_rot_t = torch_utils.slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
            
            return_dict['rg_pos_t'] = rg_pos_t
            return_dict['rg_rot_t'] = rg_rot_t
            return_dict['body_vel_t'] = body_vel_t
            return_dict['body_ang_vel_t'] = body_ang_vel_t
        
        if flags.real_traj:
            q_body_ang_vel0, q_body_ang_vel1 = self.q_gavs[f0l], self.q_gavs[f1l]
            q_rb_rot0, q_rb_rot1 = self.q_grs[f0l], self.q_grs[f1l]
            q_rg_pos0, q_rg_pos1 = self.q_gts[f0l, :], self.q_gts[f1l, :]
            q_body_vel0, q_body_vel1 = self.q_gvs[f0l], self.q_gvs[f1l]

            q_ang_vel = (1.0 - blend_exp) * q_body_ang_vel0 + blend_exp * q_body_ang_vel1
            q_rb_rot = torch_utils.slerp(q_rb_rot0, q_rb_rot1, blend_exp)
            q_rg_pos = (1.0 - blend_exp) * q_rg_pos0 + blend_exp * q_rg_pos1
            q_body_vel = (1.0 - blend_exp) * q_body_vel0 + blend_exp * q_body_vel1
            
            rg_pos[:, self.track_idx] = q_rg_pos
            rb_rot[:, self.track_idx] = q_rb_rot
            body_vel[:, self.track_idx] = q_body_vel
            body_ang_vel[:, self.track_idx] = q_ang_vel
            
        return_dict.update({
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "motion_limb_weights": self._motion_limb_weights[motion_ids],
        })
        return return_dict
        
    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, gender_betas, fix_height, mesh_parsers, masterfoot_config,  target_heading, max_len, queue, pid):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        np.random.seed(np.random.randint(5000)* pid)
        res = {}
        # print("motion_data_list[4087]:",motion_data_list[4087])
        # print("motion_data_list[4088]:",motion_data_list[4088])
        # print("motion_data_list[3932]:",motion_data_list[3932])
        assert (len(ids) == len(motion_data_list))
        for f in range(len(motion_data_list)):
            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]                     #TODO:there are something wrong with  'motion_data_list'
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]

            seq_len = curr_file['pose_aa'].shape[0] #4096
            # seq_len = curr_file['root_trans_offset'].shape[0]
            # print("curr_file['root_trans_offset'].shape[0]:",curr_file['root_trans_offset'].shape[0]) #curr_file['root_trans_offset'].shape[0]: 3
            # print("curr_file['pose_aa'].shape[0]:",curr_file['pose_aa'].shape[0])                     #curr_file['pose_aa'].shape[0]: 32

            #print("max_len:",max_len) #-1
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()
            dt = 1/curr_file['fps']
            i = 1
            while len(pose_aa.shape)!=3:
                print("$$$$$$$$$$$$$$$$$$$$$$$$\nWARNING:DUE TO LENGTH OF POSE_AA IS LESS THAN 3, RESAMPLEING!!!!!!\n$$$$$$$$$$$$$$$$$$$$$$$$")
                if(f+i>=NUM_ENVS):
                    i*=-1
                curr_file = motion_data_list[f+i]
                trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]
                pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()
                dt = 1/curr_file['fps']
                i = i + 1 if i >0 else i - 1
                
            # print("pose_aa.shape:",pose_aa.shape)
            # if (f == 3932):
            #     continue
            # print("len(motion_data_list):",len(motion_data_list))   #4096
            # print("f:",f)                                           #3932
            # print("pose_aa.shape:",pose_aa.shape)
            # print("motion_data_list[f]:",motion_data_list[f])
            B, J, N = pose_aa.shape

            ##### ZL: randomize the heading ######
            # if (not flags.im_eval) and (not flags.test):
            #     # if True:
            #     random_rot = np.zeros(3)
            #     random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
            #     random_heading_rot = sRot.from_euler("xyz", random_rot)
            #     pose_aa = pose_aa.reshape(B, -1)
            #     pose_aa[:, :3] = torch.tensor((random_heading_rot * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec())
            #     trans = torch.matmul(trans, torch.from_numpy(random_heading_rot.as_matrix().T))
            ##### ZL: randomize the heading ######
            if not target_heading is None:
                start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
                heading_inv_rot = sRot.from_quat(torch_utils.calc_heading_quat_inv(torch.from_numpy(start_root_rot.as_quat()[None, ])))
                heading_delta = sRot.from_quat(target_heading) * heading_inv_rot 
                pose_aa[:, 0] = torch.tensor((heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec())

                trans = torch.matmul(trans, torch.from_numpy(heading_delta.as_matrix().squeeze().T))


            # trans, trans_fix = MotionLibSMPL.fix_trans_height(pose_aa, trans, curr_gender_beta, mesh_parsers, fix_height_mode = fix_height)
            curr_motion = mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)
            curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items() })
            
            # if 'dof_vels' in curr_motion:
            #     print(f"Motion ID: {curr_id}, dof_vels shape: {curr_motion['dof_vels'].shape}")            
            
            res[curr_id] = (curr_file, curr_motion)
            
        if not queue is None:
            queue.put(res)
        else:
            return res


    
    