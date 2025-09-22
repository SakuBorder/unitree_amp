from pybullet_utils import bullet_client as bc
import pybullet_data as pd
import pybullet_utils.urdfEditor as ed
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mjcf',
                    help='MuJoCo xml file to be converted to URDF',
                    default='/home/dy/dy/code/PHC/phc/data/assets/robot/taihu/taihu.xml')
args = parser.parse_args()

# Initialize the Bullet client
p = bc.BulletClient()
p.setAdditionalSearchPath(pd.getDataPath())  # Ensure PyBullet's default data path is added

# Load the MuJoCo XML (MJCF) file using PyBullet
objs = p.loadMJCF(args.mjcf, flags=p.URDF_USE_IMPLICIT_CYLINDER)

# Iterate over all loaded objects (bodies)
for o in objs:
    humanoid = objs[o]
    
    # Initialize UrdfEditor from Bullet Body
    ed0 = ed.UrdfEditor()
    ed0.initializeFromBulletBody(humanoid, p._client)
    
    # Retrieve robot name and part name from Bullet Body info
    robotName = str(p.getBodyInfo(o)[1], 'utf-8')
    partName = str(p.getBodyInfo(o)[0], 'utf-8')

    # Print the robot and part names
    print("robotName =", robotName)
    print("partName =", partName)

    # Flag to control whether to save visual details (you can modify this flag based on your requirements)
    saveVisuals = False
    
    # Save the URDF file
    urdf_filename = robotName + "_" + partName + ".urdf"
    ed0.saveUrdf(urdf_filename, saveVisuals)
    print(f"URDF saved to: {urdf_filename}")
