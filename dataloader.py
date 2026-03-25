# dataloader.py (collate_fn 추가 최종본)

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

class LunarStereoDataset(data.Dataset):
    def __init__(self, data_path, mode='train', transform=True, img_height=512, img_width=512):
        # ... (이전과 동일한 __init__ 코드) ...
        super().__init__()
        self.data_path = os.path.expanduser(data_path)
        self.mode = mode
        self.apply_transform = transform
        self.img_height = img_height
        self.img_width = img_width
        
        if self.mode == 'train':
            traverses_to_load = [f"Traverse{i}" for i in range(1, 5)]
        elif self.mode == 'val':
            traverses_to_load = ["Traverse5"]
        else:
            traverses_to_load = ["Traverse6"]

        self.image_files_and_calib = []
        print(f"Loading data for mode: '{self.mode}'...")

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
                    left_images = sorted(glob.glob(os.path.join(waypoint_dir, "*camL*.png")))
                    for left_path in left_images:
                        left_filename = os.path.basename(left_path)
                        match = re.search(r'loc(\d+)_', left_filename)
                        if match:
                            loc_num_left = int(match.group(1))
                            loc_num_right = loc_num_left + 1
                            right_filename = left_filename.replace(f'loc{loc_num_left}_', f'loc{loc_num_right}_').replace('camL', 'camR')
                            right_path = os.path.join(os.path.dirname(left_path), right_filename)
                            if os.path.exists(right_path):
                                self.image_files_and_calib.append(
                                    (left_path, right_path, extrinsics_path, left_intrinsics_path, right_intrinsics_path)
                                )
        if not self.image_files_and_calib:
            print(f"ERROR: No image pairs found for mode '{self.mode}'.")
        else:
            print(f"Found {len(self.image_files_and_calib)} image pairs for mode '{self.mode}'.")

        self._calib_cache = {}

    def _load_calib_data_cv(self, extr_path, intr_l_path, intr_r_path):
        calib_data = {}
        try:
            fs_extr = cv2.FileStorage(extr_path, cv2.FILE_STORAGE_READ)
            calib_data['extrinsics'] = {'rotation_matrix': fs_extr.getNode("rotation_matrix").mat()}
            fs_extr.release()
            fs_intr_l = cv2.FileStorage(intr_l_path, cv2.FILE_STORAGE_READ)
            calib_data['left_intrinsics'] = {'camera_matrix': fs_intr_l.getNode("camera_matrix").mat()}
            fs_intr_l.release()
            fs_intr_r = cv2.FileStorage(intr_r_path, cv2.FILE_STORAGE_READ)
            calib_data['right_intrinsics'] = {'camera_matrix': fs_intr_r.getNode("camera_matrix").mat()}
            fs_intr_r.release()
        except Exception as e:
            print(f"Error loading calibration files with OpenCV: {e}")
            return None
        return calib_data

    @staticmethod
    def collate_fn(batch):
        """ ★★★ DataLoader가 사용할 사용자 정의 배치 함수 ★★★ """
        left_images, right_images, calib_data_list = zip(*batch)
        
        # 이미지들은 텐서로 묶음
        left_batch = torch.stack(left_images, 0)
        right_batch = torch.stack(right_images, 0)
        
        # 보정 데이터는 변환 없이 첫 번째 것만 사용 (배치 내에서 동일하다고 가정)
        # None이 아닌 첫 번째 유효한 데이터를 찾음
        valid_calib_data = next((item for item in calib_data_list if item is not None), None)

        return left_batch, right_batch, valid_calib_data

    def __getitem__(self, index):
        # ... (__getitem__ 코드는 이전 답변의 최종본과 동일) ...
        left_path, right_path, extr_path, intr_l_path, intr_r_path = self.image_files_and_calib[index]
        img_l = Image.open(left_path).convert('RGB')
        img_r = Image.open(right_path).convert('RGB')
        cache_key = (extr_path, intr_l_path, intr_r_path)
        if cache_key not in self._calib_cache:
            self._calib_cache[cache_key] = self._load_calib_data_cv(extr_path, intr_l_path, intr_r_path)
        calibration_data = self._calib_cache[cache_key]
        if self.apply_transform:
            if self.mode == 'train':
                resize = T.Resize((self.img_height, self.img_width), T.InterpolationMode.BICUBIC)
                img_l, img_r = resize(img_l), resize(img_r)
                if random.random() > 0.5:
                    img_l, img_r = T.functional.hflip(img_r), T.functional.hflip(img_l)
                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                    T.ColorJitter.get_params(
                        brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                        saturation=(0.8, 1.2), hue=(-0.1, 0.1)
                    )
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        img_l = T.functional.adjust_brightness(img_l, brightness_factor)
                        img_r = T.functional.adjust_brightness(img_r, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img_l = T.functional.adjust_contrast(img_l, contrast_factor)
                        img_r = T.functional.adjust_contrast(img_r, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        img_l = T.functional.adjust_saturation(img_l, saturation_factor)
                        img_r = T.functional.adjust_saturation(img_r, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        img_l = T.functional.adjust_hue(img_l, hue_factor)
                        img_r = T.functional.adjust_hue(img_r, hue_factor)
                to_tensor = T.ToTensor()
                img_l, img_r = to_tensor(img_l), to_tensor(img_r)
            else:
                transform = T.Compose([
                    T.Resize((self.img_height, self.img_width), T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                ])
                img_l, img_r = transform(img_l), transform(img_r)
        else:
            transform = T.Compose([
                T.Resize((self.img_height, self.img_width), T.InterpolationMode.BICUBIC),
                T.ToTensor()
            ])
            img_l, img_r = transform(img_l), transform(img_r)
        return img_l, img_r, calibration_data

    def __len__(self):
        return len(self.image_files_and_calib)

