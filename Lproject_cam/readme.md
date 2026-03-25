

# enhance node execute

source /opt/ros/jazzy/setup.bash
cd ~/intern/DimCam2/Lproject_cam

python3 ros2_enhancer_node.py

python3 ros2_enhancer_node.py --ros-args -p img_size:=1024

# train

conda activate dimcam_env
cd ~/intern/DimCam2/Lproject_cam
python3 train.py


# 기존 모델 이용한 노드

source /opt/ros/jazzy/setup.bash
cd ~/intern/DimCam2
python3 ros2_enhancer_node_dimcam.py