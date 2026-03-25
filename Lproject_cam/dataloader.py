# Lproject_cam/dataloader.py
# DataLoader for DimCam with FoundationStereo
# ★★★ 노출도 그룹화 버전: 같은 장면의 여러 노출도를 그룹화하고, 
#     100ms 이미지로 계산된 depth map을 모든 노출도에 적용 ★★★

import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T
import glob
import random
import re
import cv2
import numpy as np
from collections import defaultdict


class LunarStereoDataset(data.Dataset):
    """
    Lunar Stereo Dataset Loader with Exposure Grouping
    
    데이터셋 구조:
    - 각 waypoint에서 여러 노출 시간의 stereo pair 존재
    - 예: loc0_camL_001ms.png, loc0_camL_005ms.png, ..., loc0_camL_100ms.png, ...
    
    주요 기능:
    1. 같은 장면(location)의 여러 노출도 이미지를 그룹화
    2. 100ms 노출 이미지 경로를 별도로 반환 (depth map 계산용)
    3. FoundationStereo를 위한 calibration data 포함
    """
    
    # 사용 가능한 노출 시간들 (ms)
    EXPOSURE_TIMES = ['001ms', '005ms', '010ms', '025ms', '050ms', '075ms', 
                      '100ms', '150ms', '200ms', '250ms', '300ms', '350ms', 
                      '400ms', '450ms', '500ms']
    
    # Depth map 계산에 사용할 기준 노출 시간
    REFERENCE_EXPOSURE = '100ms'
    
    def __init__(self, data_path, mode='train', transform=True, 
                 img_height=512, img_width=512, 
                 use_all_exposures=True,      # ★ 모든 노출도 사용 여부
                 reference_exposure='100ms',  # ★ Depth 계산용 기준 노출
                 use_precomputed_depth=True,  # ★★★ Pre-computed depth 사용 여부
                 depth_suffix='_depth'):      # ★★★ Depth 파일 접미사
        super().__init__()
        self.data_path = os.path.expanduser(data_path)
        self.mode = mode
        self.apply_transform = transform
        self.img_height = img_height
        self.img_width = img_width
        self.use_all_exposures = use_all_exposures
        self.reference_exposure = reference_exposure
        self.use_precomputed_depth = use_precomputed_depth
        self.depth_suffix = depth_suffix
        
        if self.mode == 'train':
            traverses_to_load = [f"Traverse{i}" for i in range(1, 5)]
        elif self.mode == 'val':
            traverses_to_load = ["Traverse5"]
        else:
            traverses_to_load = ["Traverse6"]

        # ★★★ 그룹화된 이미지 파일 구조 ★★★
        # Key: (view_dir, waypoint_dir, loc_num, calib_paths)
        # Value: dict of exposure -> (left_path, right_path)
        self.exposure_groups = []
        
        print(f"Loading data for mode: '{self.mode}'...")
        print(f"Reference exposure for depth: {self.reference_exposure}")

        view_dirs = sorted(glob.glob(os.path.join(self.data_path, "View*")))
        for view_dir in view_dirs:
            view_name = os.path.basename(view_dir)
            calib_path = os.path.join(view_dir, "Calibration")
            extrinsics_path = os.path.join(calib_path, "extrinsics.yml")
            left_intrinsics_path = os.path.join(calib_path, "left_intrinsics.yml")
            right_intrinsics_path = os.path.join(calib_path, "right_intrinsics.yml")
            
            if not all(os.path.exists(p) for p in [extrinsics_path, left_intrinsics_path, right_intrinsics_path]):
                continue
                
            for traj_name in traverses_to_load:
                image_base_path = os.path.join(view_dir, view_name, traj_name)
                if not os.path.isdir(image_base_path):
                    continue
                    
                waypoint_dirs = sorted(glob.glob(os.path.join(image_base_path, "*m")))
                for waypoint_dir in waypoint_dirs:
                    # ★★★ 각 waypoint에서 노출도별로 이미지 그룹화 ★★★
                    self._process_waypoint(
                        waypoint_dir, 
                        extrinsics_path, 
                        left_intrinsics_path, 
                        right_intrinsics_path
                    )
        
        # ★★★ 그룹화된 데이터를 flat list로 변환 ★★★
        # 학습 시: 각 그룹의 모든 노출도 이미지를 개별 샘플로 처리
        self.image_files_and_calib = []
        
        for group in self.exposure_groups:
            exposures_dict = group['exposures']
            calib_paths = group['calib_paths']
            
            # 100ms 기준 이미지 경로 (depth map 계산용)
            ref_paths = exposures_dict.get(self.reference_exposure)
            if ref_paths is None:
                # 기준 노출이 없으면 이 그룹 스킵
                continue
            ref_left, ref_right = ref_paths
            
            if self.use_all_exposures:
                # 모든 노출도를 개별 샘플로 추가
                for exp_time, (left_path, right_path) in exposures_dict.items():
                    # ★★★ Pre-computed depth 경로 추가 (좌/우 모두) ★★★
                    depth_path_l = ref_left.replace('.png', f'{self.depth_suffix}.npy')
                    depth_path_r = ref_right.replace('.png', f'{self.depth_suffix}.npy')
                    
                    self.image_files_and_calib.append({
                        'left_path': left_path,
                        'right_path': right_path,
                        'ref_left_path': ref_left,    # ★ depth 계산용 200ms 이미지
                        'ref_right_path': ref_right,
                        'depth_path_l': depth_path_l if self.use_precomputed_depth else None,  # ★★★
                        'depth_path_r': depth_path_r if self.use_precomputed_depth else None,  # ★★★
                        'exposure': exp_time,
                        'calib_paths': calib_paths
                    })
            else:
                # 기준 노출만 사용 (기존 방식)
                depth_path_l = ref_left.replace('.png', f'{self.depth_suffix}.npy')
                depth_path_r = ref_right.replace('.png', f'{self.depth_suffix}.npy')
                
                self.image_files_and_calib.append({
                    'left_path': ref_left,
                    'right_path': ref_right,
                    'ref_left_path': ref_left,
                    'ref_right_path': ref_right,
                    'depth_path_l': depth_path_l if self.use_precomputed_depth else None,  # ★★★
                    'depth_path_r': depth_path_r if self.use_precomputed_depth else None,  # ★★★
                    'exposure': self.reference_exposure,
                    'calib_paths': calib_paths
                })
        
        if not self.image_files_and_calib:
            print(f"ERROR: No image pairs found for mode '{self.mode}'.")
        else:
            print(f"Found {len(self.exposure_groups)} scenes with {len(self.image_files_and_calib)} total samples for mode '{self.mode}'.")
            # 노출도별 통계 출력
            exp_counts = defaultdict(int)
            for item in self.image_files_and_calib:
                exp_counts[item['exposure']] += 1
            print(f"Exposure distribution: {dict(sorted(exp_counts.items()))}")

    def _process_waypoint(self, waypoint_dir, extr_path, intr_l_path, intr_r_path):
        """
        한 waypoint에서 모든 노출도 이미지를 그룹화
        """
        # 왼쪽 이미지 중 loc 번호별로 그룹화
        left_images = sorted(glob.glob(os.path.join(waypoint_dir, "*camL*.png")))
        
        # loc 번호별로 이미지 그룹화
        loc_groups = defaultdict(dict)
        
        for left_path in left_images:
            left_filename = os.path.basename(left_path)
            
            # loc 번호와 노출 시간 추출
            loc_match = re.search(r'loc(\d+)_', left_filename)
            exp_match = re.search(r'_(\d+ms)\.png$', left_filename)
            
            if loc_match and exp_match:
                loc_num = int(loc_match.group(1))
                exp_time = exp_match.group(1)
                
                # 대응하는 오른쪽 이미지 찾기
                loc_num_right = loc_num + 1
                right_filename = left_filename.replace(f'loc{loc_num}_', f'loc{loc_num_right}_').replace('camL', 'camR')
                right_path = os.path.join(waypoint_dir, right_filename)
                
                if os.path.exists(right_path):
                    loc_groups[loc_num][exp_time] = (left_path, right_path)
        
        # 각 loc 그룹을 exposure_groups에 추가
        for loc_num, exposures_dict in loc_groups.items():
            # 기준 노출이 있는 그룹만 추가
            if self.reference_exposure in exposures_dict:
                self.exposure_groups.append({
                    'loc_num': loc_num,
                    'waypoint': waypoint_dir,
                    'exposures': exposures_dict,
                    'calib_paths': (extr_path, intr_l_path, intr_r_path)
                })

    def _load_calib_data_cv(self, extr_path, intr_l_path, intr_r_path):
        """OpenCV를 이용한 calibration data 로드"""
        calib_data = {}
        try:
            # Extrinsics
            fs_extr = cv2.FileStorage(extr_path, cv2.FILE_STORAGE_READ)
            rotation_matrix = fs_extr.getNode("rotation_matrix").mat()
            translation_vector = fs_extr.getNode("translation_vector")
            if translation_vector is not None:
                translation_vector = translation_vector.mat()
            fs_extr.release()
            
            calib_data['extrinsics'] = {
                'rotation_matrix': rotation_matrix,
                'translation_vector': translation_vector
            }
            
            # Left intrinsics
            fs_intr_l = cv2.FileStorage(intr_l_path, cv2.FILE_STORAGE_READ)
            calib_data['left_intrinsics'] = {
                'camera_matrix': fs_intr_l.getNode("camera_matrix").mat()
            }
            fs_intr_l.release()
            
            # Right intrinsics
            fs_intr_r = cv2.FileStorage(intr_r_path, cv2.FILE_STORAGE_READ)
            calib_data['right_intrinsics'] = {
                'camera_matrix': fs_intr_r.getNode("camera_matrix").mat()
            }
            fs_intr_r.release()
            
            # Baseline 계산 (translation vector의 norm)
            if translation_vector is not None:
                calib_data['baseline'] = np.linalg.norm(translation_vector)
            else:
                # ★★★ 기본값을 실제 Lunar 데이터셋 값으로 설정 (~0.4m) ★★★
                calib_data['baseline'] = 0.4
            
            # Focal length (left camera 기준)
            K_l = calib_data['left_intrinsics']['camera_matrix']
            calib_data['focal_length'] = K_l[0, 0]  # fx
            
        except Exception as e:
            print(f"Error loading calibration files with OpenCV: {e}")
            return None
        return calib_data

    @staticmethod
    def collate_fn(batch):
        """
        DataLoader가 사용할 사용자 정의 배치 함수
        
        Returns:
            img_l_batch: 현재 노출도 왼쪽 이미지 [B, 3, H, W]
            img_r_batch: 현재 노출도 오른쪽 이미지 [B, 3, H, W]
            ref_l_batch: 200ms 노출 왼쪽 이미지 [B, 3, H, W]
            ref_r_batch: 200ms 노출 오른쪽 이미지 [B, 3, H, W]
            depth_l_batch: Pre-computed left depth map [B, 1, H, W] or None
            depth_r_batch: Pre-computed right depth map [B, 1, H, W] or None
            calib_data: Calibration 데이터
            exposure_list: 각 샘플의 노출 시간
        """
        img_l_list, img_r_list, ref_l_list, ref_r_list, depth_l_list, depth_r_list, calib_list, exp_list = zip(*batch)
        
        img_l_batch = torch.stack(img_l_list, 0)
        img_r_batch = torch.stack(img_r_list, 0)
        ref_l_batch = torch.stack(ref_l_list, 0)
        ref_r_batch = torch.stack(ref_r_list, 0)
        
        # ★★★ Left/Right Depth 배치 처리 ★★★
        if depth_l_list[0] is not None:
            depth_l_batch = torch.stack(depth_l_list, 0)
        else:
            depth_l_batch = None
            
        if depth_r_list[0] is not None:
            depth_r_batch = torch.stack(depth_r_list, 0)
        else:
            depth_r_batch = None
        
        # None이 아닌 첫 번째 유효한 데이터를 찾음
        valid_calib_data = next((item for item in calib_list if item is not None), None)

        return img_l_batch, img_r_batch, ref_l_batch, ref_r_batch, depth_l_batch, depth_r_batch, valid_calib_data, list(exp_list)

    def __getitem__(self, index):
        item = self.image_files_and_calib[index]
        
        left_path = item['left_path']
        right_path = item['right_path']
        ref_left_path = item['ref_left_path']
        ref_right_path = item['ref_right_path']
        depth_path_l = item.get('depth_path_l')  # ★★★ Pre-computed left depth 경로
        depth_path_r = item.get('depth_path_r')  # ★★★ Pre-computed right depth 경로
        exposure = item['exposure']
        extr_path, intr_l_path, intr_r_path = item['calib_paths']
        
        # 현재 노출도 이미지 로드
        img_l = Image.open(left_path).convert('RGB')
        img_r = Image.open(right_path).convert('RGB')
        
        # ★★★ 200ms 기준 이미지 로드 (depth map 계산용) ★★★
        ref_img_l = Image.open(ref_left_path).convert('RGB')
        ref_img_r = Image.open(ref_right_path).convert('RGB')
        
        # ★★★ Pre-computed Left Depth 로드 ★★★
        depth_map_l = None
        if depth_path_l and os.path.exists(depth_path_l):
            try:
                depth_np = np.load(depth_path_l).astype(np.float32)
                if depth_np.ndim == 2:
                    depth_map_l = torch.from_numpy(depth_np[np.newaxis, :, :])
                elif depth_np.ndim == 3:
                    depth_map_l = torch.from_numpy(depth_np)
                else:
                    raise ValueError(f"Unexpected depth shape: {depth_np.shape}")
            except Exception as e:
                print(f"Warning: Could not load left depth from {depth_path_l}: {e}")
                depth_map_l = None
        
        # ★★★ Pre-computed Right Depth 로드 ★★★
        depth_map_r = None
        if depth_path_r and os.path.exists(depth_path_r):
            try:
                depth_np = np.load(depth_path_r).astype(np.float32)
                if depth_np.ndim == 2:
                    depth_map_r = torch.from_numpy(depth_np[np.newaxis, :, :])
                elif depth_np.ndim == 3:
                    depth_map_r = torch.from_numpy(depth_np)
                else:
                    raise ValueError(f"Unexpected depth shape: {depth_np.shape}")
            except Exception as e:
                print(f"Warning: Could not load right depth from {depth_path_r}: {e}")
                depth_map_r = None
        
        calibration_data = self._load_calib_data_cv(extr_path, intr_l_path, intr_r_path)
        
        if self.apply_transform:
            if self.mode == 'train':
                resize = T.Resize((self.img_height, self.img_width), T.InterpolationMode.BICUBIC)
                img_l, img_r = resize(img_l), resize(img_r)
                ref_img_l, ref_img_r = resize(ref_img_l), resize(ref_img_r)
                
                # Random horizontal flip (swap left and right) - ★★★ 모든 이미지에 동일하게 적용 ★★★
                if random.random() > 0.5:
                    img_l, img_r = T.functional.hflip(img_r), T.functional.hflip(img_l)
                    ref_img_l, ref_img_r = T.functional.hflip(ref_img_r), T.functional.hflip(ref_img_l)
                    # ★★★ Depth map도 swap하고 flip ★★★
                    if depth_map_l is not None and depth_map_r is not None:
                        depth_map_l, depth_map_r = torch.flip(depth_map_r, [-1]), torch.flip(depth_map_l, [-1])
                
                # Color jitter (same for both pairs) - ★★★ ref 이미지는 color jitter 적용 안 함 ★★★
                color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                seed = torch.randint(0, 2**32, (1,)).item()
                torch.manual_seed(seed)
                img_l = color_jitter(img_l)
                torch.manual_seed(seed)
                img_r = color_jitter(img_r)
                
                to_tensor = T.ToTensor()
                img_l, img_r = to_tensor(img_l), to_tensor(img_r)
                ref_img_l, ref_img_r = to_tensor(ref_img_l), to_tensor(ref_img_r)
            else:
                transform = T.Compose([
                    T.Resize((self.img_height, self.img_width), T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                ])
                img_l, img_r = transform(img_l), transform(img_r)
                ref_img_l, ref_img_r = transform(ref_img_l), transform(ref_img_r)
        else:
            transform = T.Compose([
                T.Resize((self.img_height, self.img_width), T.InterpolationMode.BICUBIC),
                T.ToTensor()
            ])
            img_l, img_r = transform(img_l), transform(img_r)
            ref_img_l, ref_img_r = transform(ref_img_l), transform(ref_img_r)
        
        return img_l, img_r, ref_img_l, ref_img_r, depth_map_l, depth_map_r, calibration_data, exposure

    def __len__(self):
        return len(self.image_files_and_calib)

    def get_scene_count(self):
        """고유 장면(그룹) 수 반환"""
        return len(self.exposure_groups)


# --- Legacy compatibility: 기존 방식으로 사용 가능한 Wrapper ---

class LunarStereoDatasetLegacy(LunarStereoDataset):
    """
    기존 dataloader와 동일한 인터페이스 (호환성 유지)
    100ms 노출 이미지만 사용
    """
    def __init__(self, data_path, mode='train', transform=True, 
                 img_height=512, img_width=512):
        super().__init__(
            data_path, mode, transform, img_height, img_width,
            use_all_exposures=False,  # 기준 노출만 사용
            reference_exposure='100ms'
        )
    
    @staticmethod
    def collate_fn(batch):
        """기존 형식과 동일한 출력"""
        img_l_list, img_r_list, ref_l_list, ref_r_list, calib_list, exp_list = zip(*batch)
        
        # ref 이미지를 main 이미지로 사용 (100ms만 사용하므로 동일)
        img_l_batch = torch.stack(img_l_list, 0)
        img_r_batch = torch.stack(img_r_list, 0)
        
        valid_calib_data = next((item for item in calib_list if item is not None), None)
        
        return img_l_batch, img_r_batch, valid_calib_data

    def __getitem__(self, index):
        img_l, img_r, ref_l, ref_r, calib, exp = super().__getitem__(index)
        # 기존 형식으로 반환
        return img_l, img_r, calib


# --- StereoDatasetWithIntrinsics (FoundationStereo 호환) ---

class StereoDatasetWithIntrinsics(LunarStereoDataset):
    """
    FoundationStereo 호환 데이터셋
    - intrinsics.txt 파일 형식 지원 (K matrix + baseline)
    """
    def get_intrinsics_for_foundation_stereo(self, index):
        """
        FoundationStereo 형식의 intrinsics 반환
        Returns:
            K: 3x3 camera matrix (flattened to 9 values)
            baseline: float (meters)
        """
        item = self.image_files_and_calib[index]
        extr_path, intr_l_path, _ = item['calib_paths']
        calib_data = self._load_calib_data_cv(extr_path, intr_l_path, intr_l_path)
        
        if calib_data is None:
            return None, None
        
        K = calib_data['left_intrinsics']['camera_matrix']
        baseline = calib_data['baseline']
        
        return K.flatten(), baseline
