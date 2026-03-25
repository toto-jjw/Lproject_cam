#!/usr/bin/env python3
# Lproject_sim/src/nodes/camera_noise_node.py
"""
Camera Noise Node

ROS2 이미지 토픽에 센서 노이즈를 추가하는 노드입니다.
원본 토픽을 구독하고 노이즈가 적용된 이미지를 새 토픽으로 발행합니다.

지원하는 노이즈 유형:
- Gaussian noise (가우시안 노이즈)
- Salt & Pepper noise (점 노이즈)
- Exposure variation (노출 변화)
- Motion blur (모션 블러) - TODO

Usage:
    ros2 run <package> camera_noise_node --ros-args -p config_file:=/path/to/config.yaml
    
    또는 launch 파일에서:
    ros2 launch <package> camera_noise.launch.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import numpy as np
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# 센서 데이터에 적합한 QoS 프로파일
# Subscriber: BEST_EFFORT (원본 센서 데이터 수신)
SENSOR_SUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE
)

# Publisher: RELIABLE (메시지 손실 방지)
SENSOR_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE
)


@dataclass
class NoiseConfig:
    """노이즈 설정 데이터 클래스"""
    enabled: bool = True
    
    # Gaussian Noise
    gaussian_enabled: bool = True
    gaussian_mean: float = 0.0
    gaussian_std: float = 5.0  # 픽셀 값 기준 (0-255 스케일)
    
    # Salt & Pepper Noise
    salt_pepper_enabled: bool = False
    salt_pepper_prob: float = 0.001  # 각 픽셀이 노이즈가 될 확률
    
    # Exposure Variation
    exposure_enabled: bool = False
    exposure_variation: float = 0.1  # 밝기 변화 비율 (±10%)
    
    # Depth Noise (depth 이미지 전용)
    depth_gaussian_std: float = 0.01  # 미터 단위
    depth_dropout_prob: float = 0.001  # 깊이 값 손실 확률


@dataclass  
class TopicConfig:
    """토픽별 설정"""
    input_topic: str
    output_topic: str
    image_type: str = "rgb"  # "rgb" or "depth"
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)


class CameraNoiseNode(Node):
    """
    카메라 이미지에 센서 노이즈를 추가하는 ROS2 노드
    """
    
    # 기본 설정 파일 경로
    DEFAULT_CONFIG_FILE = '/home/jaewon/Lproject_sim/config/simulation_config.yaml'
    
    def __init__(self):
        super().__init__('camera_noise_node')
        
        # CV Bridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()
        
        # 성능 측정용
        self._process_times: Dict[str, list] = {}
        self._frame_counts: Dict[str, int] = {}
        self._last_log_time = self.get_clock().now()
        
        # Declare parameters
        self.declare_parameter('config_file', self.DEFAULT_CONFIG_FILE)
        self.declare_parameter('enabled', True)
        
        # 개별 노이즈 파라미터 (config 파일 없이 사용 가능)
        self.declare_parameter('gaussian_enabled', True)
        self.declare_parameter('gaussian_mean', 0.0)
        self.declare_parameter('gaussian_std', 5.0)
        self.declare_parameter('salt_pepper_enabled', False)
        self.declare_parameter('salt_pepper_prob', 0.001)
        self.declare_parameter('exposure_enabled', False)
        self.declare_parameter('exposure_variation', 0.1)
        self.declare_parameter('depth_gaussian_std', 0.01)
        self.declare_parameter('depth_dropout_prob', 0.001)
        
        # 토픽 설정 파라미터
        self.declare_parameter('rgb_topics', [
            '/stereo/left/rgb',
            '/stereo/right/rgb'
        ])
        self.declare_parameter('depth_topics', [
            '/front_camera/depth/depth'
        ])
        self.declare_parameter('output_suffix', '_noisy')
        
        # 설정 로드
        self.load_config()
        
        # Publishers & Subscribers 초기화
        self._subs: Dict[str, Any] = {}
        self._pubs: Dict[str, Any] = {}
        
        # 고정 노이즈 마스크 (토픽별로 저장)
        # Salt & Pepper 노이즈의 위치를 고정하기 위해 사용
        self._salt_masks: Dict[str, np.ndarray] = {}
        self._pepper_masks: Dict[str, np.ndarray] = {}
        self._depth_dropout_masks: Dict[str, np.ndarray] = {}
        self._noise_buffers: Dict[str, Any] = {}  # 노이즈 생성 최적화용 버퍼
        
        if self.enabled:
            self.setup_topics()
            self.get_logger().info(f"Camera Noise Node initialized with {len(self.topic_configs)} topics")
        else:
            self.get_logger().info("Camera Noise Node disabled")
    
    def load_config(self):
        """설정 파일 또는 파라미터에서 설정 로드"""
        config_file = self.get_parameter('config_file').value
        
        self.enabled = self.get_parameter('enabled').value
        
        # 기본 노이즈 설정
        self.default_noise_config = NoiseConfig(
            enabled=self.enabled,
            gaussian_enabled=self.get_parameter('gaussian_enabled').value,
            gaussian_mean=self.get_parameter('gaussian_mean').value,
            gaussian_std=self.get_parameter('gaussian_std').value,
            salt_pepper_enabled=self.get_parameter('salt_pepper_enabled').value,
            salt_pepper_prob=self.get_parameter('salt_pepper_prob').value,
            exposure_enabled=self.get_parameter('exposure_enabled').value,
            exposure_variation=self.get_parameter('exposure_variation').value,
            depth_gaussian_std=self.get_parameter('depth_gaussian_std').value,
            depth_dropout_prob=self.get_parameter('depth_dropout_prob').value,
        )
        
        self.topic_configs: list[TopicConfig] = []
        
        # Config 파일에서 로드 시도
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    
                env_config = yaml_config.get('environment', {})
                noise_config = env_config.get('camera_noise', {})
                
                self.enabled = noise_config.get('enabled', self.enabled)
                
                # YAML에서 노이즈 설정 오버라이드
                if 'gaussian' in noise_config:
                    g = noise_config['gaussian']
                    self.default_noise_config.gaussian_enabled = g.get('enabled', True)
                    self.default_noise_config.gaussian_mean = g.get('mean', 0.0)
                    self.default_noise_config.gaussian_std = g.get('std', 5.0)
                    
                if 'salt_pepper' in noise_config:
                    sp = noise_config['salt_pepper']
                    self.default_noise_config.salt_pepper_enabled = sp.get('enabled', False)
                    self.default_noise_config.salt_pepper_prob = sp.get('prob', 0.001)
                    
                if 'exposure' in noise_config:
                    exp = noise_config['exposure']
                    self.default_noise_config.exposure_enabled = exp.get('enabled', False)
                    self.default_noise_config.exposure_variation = exp.get('variation', 0.1)
                    
                if 'depth' in noise_config:
                    d = noise_config['depth']
                    self.default_noise_config.depth_gaussian_std = d.get('gaussian_std', 0.01)
                    self.default_noise_config.depth_dropout_prob = d.get('dropout_prob', 0.001)
                
                # 토픽 설정
                if 'topics' in noise_config:
                    topics = noise_config['topics']
                    for t in topics.get('rgb', []):
                        self.topic_configs.append(TopicConfig(
                            input_topic=t,
                            output_topic=t + noise_config.get('output_suffix', '_noisy'),
                            image_type='rgb',
                            noise_config=self.default_noise_config
                        ))
                    for t in topics.get('depth', []):
                        self.topic_configs.append(TopicConfig(
                            input_topic=t,
                            output_topic=t + noise_config.get('output_suffix', '_noisy'),
                            image_type='depth',
                            noise_config=self.default_noise_config
                        ))
                        
                self.get_logger().info(f"Loaded config from: {config_file}")
                
            except Exception as e:
                self.get_logger().warn(f"Failed to load config file: {e}, using parameters")
        
        # 파라미터에서 토픽 설정 (config 파일 없거나 토픽 설정이 없을 때)
        if not self.topic_configs:
            output_suffix = self.get_parameter('output_suffix').value
            
            rgb_topics = self.get_parameter('rgb_topics').value
            for t in rgb_topics:
                self.topic_configs.append(TopicConfig(
                    input_topic=t,
                    output_topic=t + output_suffix,
                    image_type='rgb',
                    noise_config=self.default_noise_config
                ))
                
            depth_topics = self.get_parameter('depth_topics').value
            for t in depth_topics:
                self.topic_configs.append(TopicConfig(
                    input_topic=t,
                    output_topic=t + output_suffix,
                    image_type='depth',
                    noise_config=self.default_noise_config
                ))
    
    def setup_topics(self):
        """토픽 구독자/발행자 설정"""
        for config in self.topic_configs:
            # Publisher 생성 - Raw Image (RELIABLE QoS - 메시지 손실 방지)
            self._pubs[config.input_topic] = self.create_publisher(
                Image, 
                config.output_topic, 
                SENSOR_PUB_QOS
            )
            
            # Publisher 생성 - Compressed Image (RGB만)
            if config.image_type == 'rgb':
                compressed_topic = config.output_topic + '/compressed'
                self._pubs[config.input_topic + '_compressed'] = self.create_publisher(
                    CompressedImage,
                    compressed_topic,
                    SENSOR_PUB_QOS
                )
            
            # Subscriber 생성 (BEST_EFFORT QoS - 센서 데이터 수신)
            self._subs[config.input_topic] = self.create_subscription(
                Image,
                config.input_topic,
                lambda msg, cfg=config: self.image_callback(msg, cfg),
                SENSOR_SUB_QOS
            )
            
            if config.image_type == 'rgb':
                self.get_logger().info(
                    f"  {config.input_topic} -> {config.output_topic} + /compressed ({config.image_type})"
                )
            else:
                self.get_logger().info(
                    f"  {config.input_topic} -> {config.output_topic} ({config.image_type})"
                )
    
    def image_callback(self, msg: Image, config: TopicConfig):
        """이미지 콜백 - 노이즈 적용 후 발행"""
        import time
        start_time = time.time()
        
        if not config.noise_config.enabled:
            # 노이즈 비활성화 시 원본 그대로 발행
            self._pubs[config.input_topic].publish(msg)
            return
        
        try:
            # ROS Image -> NumPy array (zero-copy when possible)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            if config.image_type == 'rgb':
                noisy_image = self.apply_rgb_noise_fast(cv_image, config.noise_config, config.input_topic)
            else:  # depth
                noisy_image = self.apply_depth_noise_fast(cv_image, config.noise_config, config.input_topic)
            
            noisy_msg = self.bridge.cv2_to_imgmsg(noisy_image, encoding=msg.encoding)
            
            # 타임스탬프 유지
            noisy_msg.header = msg.header
            
            # Raw Image 발행
            self._pubs[config.input_topic].publish(noisy_msg)
            
            # Compressed Image 발행 (RGB만)
            if config.image_type == 'rgb':
                compressed_key = config.input_topic + '_compressed'
                if compressed_key in self._pubs:
                    # BGR로 변환 (cv2.imencode 용)
                    if len(noisy_image.shape) == 3:
                        image_for_encode = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR) if noisy_image.shape[2] == 3 else noisy_image
                    else:
                        image_for_encode = noisy_image
                    
                    # JPEG 압축
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    _, compressed_data = cv2.imencode('.jpg', image_for_encode, encode_param)
                    
                    compressed_msg = CompressedImage()
                    compressed_msg.header = msg.header
                    compressed_msg.format = 'jpeg'
                    compressed_msg.data = compressed_data.tobytes()
                    self._pubs[compressed_key].publish(compressed_msg)
            
            # 성능 측정
            process_time = time.time() - start_time
            topic = config.input_topic
            if topic not in self._process_times:
                self._process_times[topic] = []
                self._frame_counts[topic] = 0
            self._process_times[topic].append(process_time)
            self._frame_counts[topic] += 1
            
            # 5초마다 로깅
            now = self.get_clock().now()
            if (now - self._last_log_time).nanoseconds > 5e9:
                for t, times in self._process_times.items():
                    if times:
                        avg_ms = sum(times) / len(times) * 1000
                        fps = self._frame_counts[t] / 5.0
                        self.get_logger().info(
                            f'[{t}] Process: {avg_ms:.1f}ms, Output: {fps:.1f} FPS'
                        )
                self._process_times = {t: [] for t in self._process_times}
                self._frame_counts = {t: 0 for t in self._frame_counts}
                self._last_log_time = now
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def apply_rgb_noise_fast(self, image: np.ndarray, config: NoiseConfig, topic: str) -> np.ndarray:
        """RGB 이미지에 노이즈 적용 (최적화 버전)"""
        # int16으로 작업 (float32보다 2배 빠름, overflow 방지)
        noisy = image.astype(np.int16)
        
        # 1. Gaussian Noise (매 프레임 랜덤) - 정수 노이즈로 최적화
        if config.gaussian_enabled and config.gaussian_std > 0:
            # 미리 계산된 노이즈 버퍼 재사용 (첫 프레임에서 shape 캐시)
            noise_key = f"{topic}_gaussian"
            if noise_key not in self._noise_buffers:
                self._noise_buffers[noise_key] = image.shape
            
            # int16 노이즈 생성 (float보다 빠름)
            gaussian = np.random.randint(
                int(-config.gaussian_std * 3),
                int(config.gaussian_std * 3) + 1,
                size=image.shape,
                dtype=np.int16
            )
            noisy += gaussian
        
        # 2. Salt & Pepper Noise (고정 위치)
        if config.salt_pepper_enabled and config.salt_pepper_prob > 0:
            if topic not in self._salt_masks:
                self._salt_masks[topic] = np.random.random(image.shape[:2]) < config.salt_pepper_prob / 2
                self._pepper_masks[topic] = np.random.random(image.shape[:2]) < config.salt_pepper_prob / 2
                self.get_logger().info(
                    f"[{topic}] Created fixed salt/pepper masks: "
                    f"salt={np.sum(self._salt_masks[topic])}, pepper={np.sum(self._pepper_masks[topic])}"
                )
            noisy[self._salt_masks[topic]] = 255
            noisy[self._pepper_masks[topic]] = 0
        
        # 3. Exposure Variation (매 프레임 랜덤)
        if config.exposure_enabled and config.exposure_variation > 0:
            factor = 1.0 + np.random.uniform(-config.exposure_variation, config.exposure_variation)
            noisy = (noisy * factor).astype(np.int16)
        
        # Clip and convert back (in-place where possible)
        np.clip(noisy, 0, 255, out=noisy)
        return noisy.astype(np.uint8)
    
    def apply_depth_noise_fast(self, image: np.ndarray, config: NoiseConfig, topic: str) -> np.ndarray:
        """Depth 이미지에 노이즈 적용 (최적화 버전)"""
        is_uint16 = image.dtype == np.uint16
        
        if is_uint16:
            noisy = image.astype(np.int32)  # overflow 방지
            scale = 1000.0
        else:
            noisy = image.copy()
            scale = 1.0
        
        # 1. Gaussian Noise (단순화 - 균일 노이즈로 대체, 더 빠름)
        if config.depth_gaussian_std > 0:
            noise_mm = int(config.depth_gaussian_std * scale * 3)  # 3-sigma
            if noise_mm > 0:
                gaussian = np.random.randint(-noise_mm, noise_mm + 1, size=image.shape, dtype=np.int32)
                noisy += gaussian
        
        # 2. Depth Dropout (고정 위치)
        if config.depth_dropout_prob > 0:
            if topic not in self._depth_dropout_masks:
                self._depth_dropout_masks[topic] = np.random.random(image.shape) < config.depth_dropout_prob
                self.get_logger().info(f"[{topic}] Created fixed depth dropout mask")
            noisy[self._depth_dropout_masks[topic]] = 0
        
        # Clip and convert back
        if is_uint16:
            np.clip(noisy, 0, 65535, out=noisy)
            return noisy.astype(np.uint16)
        else:
            np.clip(noisy, 0, None, out=noisy)
            return noisy.astype(np.float32)
    
    # Legacy methods (kept for compatibility)
    def apply_rgb_noise(self, image: np.ndarray, config: NoiseConfig, topic: str) -> np.ndarray:
        """RGB 이미지에 노이즈 적용 (레거시 - apply_rgb_noise_fast 사용 권장)"""
        return self.apply_rgb_noise_fast(image, config, topic)
    
    def apply_depth_noise(self, image: np.ndarray, config: NoiseConfig, topic: str) -> np.ndarray:
        """Depth 이미지에 노이즈 적용 (레거시 - apply_depth_noise_fast 사용 권장)"""
        return self.apply_depth_noise_fast(image, config, topic)


def main(args=None):
    rclpy.init(args=args)
    
    node = CameraNoiseNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
