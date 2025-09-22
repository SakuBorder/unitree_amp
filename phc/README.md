# RobotRetarget
python scripts/vis/vis_q_mj_taihu.py robot=taihu +motion_name="0-Male2Walking_c3d_B15 -  Walk turn around_poses"
python scripts/data_process/fit_smpl_motion.py robot=taihu +amass_root=/home/dy/dy/code/PHC/data/AMASS/AMASS_Complete/ACCAD
python scripts/data_process/grad_fit_taihu_shape.py 

python scripts/data_process/fit_smpl_motion.py robot=taihu_12dof +amass_root=/home/dy/dy/code/PHC/data/AMASS/AMASS_Complete/ACCAD
python scripts/data_process/fit_smpl_shape.py robot=ti5robot
python scripts/vis/vis_q_mj_taihu.py robot=taihu_12dof +motion_name="0-Female1Walking_c3d_B3 - walk1_poses"
python script/vis/vis_taihu.py