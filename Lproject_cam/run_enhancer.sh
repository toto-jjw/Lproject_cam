#!/bin/bash
# DimCam Enhancer ROS2 Node 실행 스크립트
# 사용법: ./run_enhancer.sh [model_weights_path]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 기본 가중치 경로
DEFAULT_WEIGHTS="${SCRIPT_DIR}/snapshots_foundation_stereo/dimcam_enhancer_best.pth"
WEIGHTS_PATH="${1:-$DEFAULT_WEIGHTS}"

# Conda 환경 활성화 (필요시)
# NOTE: ROS 2 (Jazzy/Humble)는 시스템 Python 버전과 일치해야 rclpy가 동작합니다.
# Conda 환경(Python 3.11)과 ROS 2(Python 3.12/3.10) 버전이 다를 경우 충돌이 발생하므로
# 아래 Conda 활성화 부분은 주석 처리합니다. 시스템 Python에 torch를 설치해서 사용하세요.
#
# if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
#     source "$HOME/anaconda3/etc/profile.d/conda.sh"
#     conda activate dimcam_env
# fi

# ROS 2 환경 소싱 (필요시)
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
elif [ -f "/opt/ros/iron/setup.bash" ]; then
    source /opt/ros/iron/setup.bash
fi

echo "=============================================="
echo "DimCam Enhancer ROS2 Node"
echo "=============================================="
echo "Model weights: $WEIGHTS_PATH"
echo ""

# 노드 실행
cd "$SCRIPT_DIR"
python3 ros2_enhancer_node.py --ros-args \
    -p model_weights:="$WEIGHTS_PATH" \
    -p device:="cuda" \
    -p img_size:=512 \
    -p use_fp16:=true \
    -p publish_rate:=20.0
