import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import torchvision.transforms as T 

random.seed(1143)

def populate_train_list(lowlight_images_path):
    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_list = []
    for ext in supported_extensions:
        image_list.extend(glob.glob(os.path.join(lowlight_images_path, ext)))
    
    random.shuffle(image_list)
    return image_list


class lowlight_loader(data.Dataset):

    def __init__(self, lowlight_images_path, size=512):
        """
        Args:
            lowlight_images_path (str): 학습 이미지가 있는 폴더 경로.
            size (int): 최종적으로 잘라낼 이미지의 크기 (정사각형).
        """
        self.train_list = populate_train_list(lowlight_images_path) 
        self.size = size
        self.data_list = self.train_list
        print("Total training examples found:", len(self.train_list))

        self.transform = T.Compose([
            T.Resize(size),
            T.RandomCrop(size),
            T.ToTensor(),
        ])
        # ------------------------------------

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        
        try:
            img = Image.open(data_lowlight_path).convert('RGB')
                
        except Exception as e:
            print(f"Error opening or converting image {data_lowlight_path}: {e}")
            return torch.zeros(3, self.size, self.size)

        return self.transform(img)
        # ------------------------------------

    def __len__(self):
        return len(self.data_list)
