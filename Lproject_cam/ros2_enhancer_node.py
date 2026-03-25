#!/usr/bin/env python3
"""
ROS2 Stereo Image Enhancement Node
DimCam Enhancer를 사용하여 스테레오 이미지를 실시간으로 향상시킵니다.

Subscribed Topics:
    /stereo/left/rgb (sensor_msgs/Image): Left camera RGB image
    /stereo/right/rgb (sensor_msgs/Image): Right camera RGB image
    /stereo/left/camera_info (sensor_msgs/CameraInfo): Left camera info
    /stereo/right/camera_info (sensor_msgs/CameraInfo): Right camera info
    /tf (tf2_msgs/TFMessage): Camera transforms

Published Topics:
    /stereo/left/enhanced (sensor_msgs/Image): Enhanced left image
    /stereo/right/enhanced (sensor_msgs/Image): Enhanced right image
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
from threading import Lock
import time

# 모델 경로 추가
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, '..'))


class DimCamEnhancerNode(Node):
    """DimCam Stereo Image Enhancement ROS2 Node"""
    
    def __init__(self):
        super().__init__('dimcam_enhancer_node')
        
        # --- 파라미터 선언 ---
        self.declare_parameter('model_weights', 
            os.path.join(SCRIPT_DIR, 'snapshots_foundation_stereo/dimcam_enhancer_best.pth'))
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('img_size', 512)
        self.declare_parameter('gamma_min', 0.001)
        self.declare_parameter('gamma_max', 100.0)
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('max_rate', 15.0)  # 목표 처리 Hz (65ms 추론 → 약 15Hz 가능)
        self.declare_parameter('sync_slop', 0.1)  # 시간 동기화 허용 오차 (초)
        
        # 파라미터 로드
        self.model_weights = self.get_parameter('model_weights').value
        self.device_name = self.get_parameter('device').value
        self.img_size = self.get_parameter('img_size').value
        self.gamma_min = self.get_parameter('gamma_min').value
        self.gamma_max = self.get_parameter('gamma_max').value
        self.use_fp16 = self.get_parameter('use_fp16').value
        self.max_rate = self.get_parameter('max_rate').value
        self.sync_slop = self.get_parameter('sync_slop').value
        
        # --- 디바이스 설정 ---
        if self.device_name == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.get_logger().info(f'Using GPU: {torch.cuda.get_device_name(0)}')
        else:
            self.device = torch.device('cpu')
            self.get_logger().info('Using CPU')
        
        # --- 모델 로드 ---
        self.model = None
        self.model_lock = Lock()
        self._load_model()
        
        # --- CV Bridge ---
        self.bridge = CvBridge()
        
        # --- 처리 상태 ---
        self.process_count = 0
        self.start_time = time.time()
        self.last_process_time = 0.0  # Rate limiting용
        self.left_camera_info = None
        self.right_camera_info = None
        
        # --- ★ 최신 이미지 버퍼 ---
        self.latest_left_msg = None
        self.image_lock = Lock()
        self.is_processing = False  # 처리 중 플래그
        self.last_processed_stamp = None  # 마지막 처리한 타임스탬프 (중복 방지)
        
        # --- TF2 버퍼 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # --- QoS 설정 (카메라 토픽용) ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # ★ 최신 1개만 유지
        )
        
        # --- ★★★ 단순한 Subscriber (직접 처리 방식) ★★★ ---
        # Left 이미지: 버퍼링만
        self.left_image_sub = self.create_subscription(
            Image, '/stereo/left/rgb', self.left_image_callback, sensor_qos)
        # Right 이미지: 여기서 처리 트리거
        self.right_image_sub = self.create_subscription(
            Image, '/stereo/right/rgb', self.right_image_callback, sensor_qos)
        
        # Camera info subscribers
        self.left_info_sub = self.create_subscription(
            CameraInfo, '/stereo/left/camera_info', self.left_info_callback, sensor_qos)
        self.right_info_sub = self.create_subscription(
            CameraInfo, '/stereo/right/camera_info', self.right_info_callback, sensor_qos)
        
        # --- Publishers (RELIABLE QoS로 설정하여 ros2 topic hz와 호환) ---
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.left_enhanced_pub = self.create_publisher(Image, '/stereo/left/enhanced', pub_qos)
        self.right_enhanced_pub = self.create_publisher(Image, '/stereo/right/enhanced', pub_qos)
        
        self.get_logger().info('DimCam Enhancer Node initialized (Direct processing)')
        self.get_logger().info(f'  - Model weights: {self.model_weights}')
        self.get_logger().info(f'  - Image size: {self.img_size}x{self.img_size}')
        self.get_logger().info(f'  - Target rate: {self.max_rate} Hz')
        self.get_logger().info(f'  - FP16 inference: {self.use_fp16}')
    
    def _load_model(self):
        """DimCam 모델 로드 (FoundationStereo 제외)"""
        try:
            # 모델 임포트 (FoundationStereo 없이)
            import model as dimcam_model
            
            self.get_logger().info('Loading DimCam model...')
            
            # 모델 초기화 (depth 비활성화 - FoundationStereo 불필요)
            self.model = dimcam_model.DimCamEnhancer(
                img_size=self.img_size,
                embed_dim=64,
                num_blocks=5,
                lambda_depth=0.0,  # ★ Depth 비활성화 (FoundationStereo 불필요)
                use_grayscale=True,
                residual_scale=1.0,
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
                depth_mode='midas',  # 사용 안 함 (lambda_depth=0)
            ).to(self.device)
            
            # 가중치 로드
            if os.path.exists(self.model_weights):
                state_dict = torch.load(self.model_weights, map_location=self.device)
                
                # depth_net 관련 가중치 제거 (로드 시 불일치 방지)
                filtered_state_dict = {
                    k: v for k, v in state_dict.items() 
                    if not k.startswith('depth_net')
                }
                self.model.load_state_dict(filtered_state_dict, strict=False)
                self.get_logger().info(f'Loaded model weights from {self.model_weights}')
            else:
                self.get_logger().warn(f'Model weights not found: {self.model_weights}')
                self.get_logger().warn('Using randomly initialized weights!')
            
            self.model.eval()
            
            # FP16 변환
            if self.use_fp16 and self.device.type == 'cuda':
                self.model = self.model.half()
                self.get_logger().info('Model converted to FP16')
            
            self.get_logger().info('Model loaded successfully!')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise
    
    def left_image_callback(self, msg: Image):
        """Left image callback - 최신 이미지만 저장"""
        with self.image_lock:
            self.latest_left_msg = msg
    
    def right_image_callback(self, msg: Image):
        
        # Rate limiting: 최소 간격 체크
        current_time = time.time()
        min_interval = 1.0 / self.max_rate
        if (current_time - self.last_process_time) < min_interval:
            return  # 너무 빠름
        
        # 이미 처리 중이면 스킵
        if self.is_processing:
            return
        
        # Left 이미지 확인
        with self.image_lock:
            if self.latest_left_msg is None:
                return
            left_msg = self.latest_left_msg
        
        right_msg = msg
        
        # 타임스탬프 중복 체크
        current_stamp = right_msg.header.stamp.sec + right_msg.header.stamp.nanosec * 1e-9
        if self.last_processed_stamp is not None:
            if abs(current_stamp - self.last_processed_stamp) < 0.001:
                return
        
        # ★ 처리 시작
        self.is_processing = True
        self.last_process_time = current_time
        
        try:
            start_time = time.time()
            
            # 이미지 텐서 변환
            left_tensor, left_size = self._msg_to_tensor(left_msg)
            right_tensor, right_size = self._msg_to_tensor(right_msg)
            
            # CUDA 동기화
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            inference_start = time.time()
            
            # 모델 추론
            with self.model_lock:
                with torch.no_grad():
                    if self.use_fp16 and self.device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(left_tensor, right_tensor)
                    else:
                        outputs = self.model(left_tensor, right_tensor)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    enhanced_left = outputs[0].clamp(0, 1)
                    enhanced_right = outputs[1].clamp(0, 1)
            
            inference_time = time.time() - inference_start
            
            # ROS 메시지 변환 및 퍼블리시
            left_enhanced_msg = self._tensor_to_msg(
                enhanced_left, left_size, left_msg.header)
            right_enhanced_msg = self._tensor_to_msg(
                enhanced_right, right_size, right_msg.header)
            
            self.left_enhanced_pub.publish(left_enhanced_msg)
            self.right_enhanced_pub.publish(right_enhanced_msg)
            
            self.last_processed_stamp = current_stamp
            self.process_count += 1
            
            # 성능 로깅 (5초마다)
            process_time = time.time() - start_time
            elapsed = current_time - self.start_time
            if elapsed > 5.0 and self.process_count > 0:
                avg_fps = self.process_count / elapsed
                self.get_logger().info(
                    f'Inference: {inference_time*1000:.1f}ms, Total: {process_time*1000:.1f}ms, '
                    f'Output: {avg_fps:.1f} FPS'
                )
                self.start_time = current_time
                self.process_count = 0
                
        except Exception as e:
            self.get_logger().error(f'Error processing images: {e}')
        finally:
            self.is_processing = False
    
    def left_info_callback(self, msg: CameraInfo):
        """Left camera info subscriber callback"""
        self.left_camera_info = msg
    
    def right_info_callback(self, msg: CameraInfo):
        """Right camera info subscriber callback"""
        self.right_camera_info = msg
    
    def _msg_to_tensor(self, msg: Image) -> torch.Tensor:
        """ROS Image 메시지를 PyTorch 텐서로 변환"""
        # CV Bridge로 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        
        # 원본 크기 저장
        original_h, original_w = cv_image.shape[:2]
        
        # 리사이즈
        cv_image = cv2.resize(cv_image, (self.img_size, self.img_size))
        
        # 정규화 [0, 1]
        image = cv_image.astype(np.float32) / 255.0
        
        # CHW 형식으로 변환
        image = np.transpose(image, (2, 0, 1))
        
        # 텐서 변환
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        
        if self.use_fp16 and self.device.type == 'cuda':
            tensor = tensor.half()
        
        return tensor, (original_h, original_w)
    
    def _tensor_to_msg(self, tensor: torch.Tensor, original_size: tuple, 
                       header) -> Image:
        """PyTorch 텐서를 ROS Image 메시지로 변환"""
        # CPU로 이동 및 float32 변환
        image = tensor.squeeze(0).float().cpu().numpy()
        
        # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
        
        # [0, 1] -> [0, 255]
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        
        # 원본 크기로 리사이즈
        original_h, original_w = original_size
        image = cv2.resize(image, (original_w, original_h))
        
        # ROS 메시지로 변환
        msg = self.bridge.cv2_to_imgmsg(image, encoding='rgb8')
        msg.header = header
        
        return msg
    
    def get_camera_transform(self, target_frame: str, source_frame: str):
        """TF에서 카메라 변환 정보 획득"""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time())
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return None


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = DimCamEnhancerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
