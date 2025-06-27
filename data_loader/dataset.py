import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import config
from sklearn.model_selection import train_test_split

class DeepfakeDataset(Dataset):
    def __init__(self, dataset_paths=None, mode='train', split='train', samples=None):
        super(DeepfakeDataset, self).__init__()
        self.mode = mode
        self.split = split
        self.num_frames = config.NUM_FRAMES

        # 设置训练与测试不同的增强
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(config.TRAIN_TRANSFORM['resize']),  # 例如 (112, 112)
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.RandomHorizontalFlip(),  # 默认概率 0.5
                # transforms.RandomRotation(10),  # 旋转角度限定在 ±10°
                # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # 小范围平移与缩放
                # 如果细节重要，建议暂时移除模糊，或者使用较小的 sigma
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5321, 0.4181, 0.3980],
                                     std=[0.2643, 0.2185, 0.2210])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(config.TEST_TRANSFORM['resize']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5321, 0.4181, 0.3980],
                                     std=[0.2643, 0.2185, 0.2210])
            ])

        if samples is not None:
            self.samples = samples
        else:
            dataset_paths = config.TRAIN_DATASET_PATHS if self.mode == 'train' else config.TEST_DATASET_PATHS
            full_samples = []
            for ds_name, ds_path in dataset_paths.items():
                for label_str in ['REAL', 'FAKE']:
                    label = 0 if label_str == 'REAL' else 1
                    folder = os.path.join(ds_path, label_str)
                    if not os.path.exists(folder):
                        continue
                    video_names = sorted(os.listdir(folder))
                    for video_name in video_names:
                        video_folder = os.path.join(folder, video_name)
                        if os.path.isdir(video_folder):
                            full_samples.append({'video_path': video_folder, 'label': label})
            random.shuffle(full_samples)
            if self.mode == 'train' and self.split in ['train', 'val']:
                split_ratio = config.TRAIN_VAL_SPLIT_RATIO
                train_samples, val_samples = train_test_split(full_samples, test_size=(1 - split_ratio), random_state=42)
                self.samples = train_samples if self.split == 'train' else val_samples
            else:
                self.samples = full_samples

    def __len__(self):
        return len(self.samples)

    def _load_video_frames(self, video_folder):
        # 获取视频文件夹内所有帧文件（支持 jpg、png、jpeg）
        frame_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        frame_files = sorted(frame_files)
        num_available = len(frame_files)

        if num_available >= self.num_frames:
            indices = np.linspace(0, num_available - 1, self.num_frames, dtype=int)
        else:
            indices = list(range(num_available))
            indices += [num_available - 1] * (self.num_frames - num_available)
        selected_files = [frame_files[i] for i in indices]

        # 为整个视频生成固定随机种子，保证所有帧增强一致
        video_seed = random.randint(0, 10000)
        random.seed(video_seed)
        torch.manual_seed(video_seed)

        frames = []
        for file in selected_files:
            img = Image.open(file).convert('RGB')
            # 对整个视频使用相同的随机增强
            img = self.transform(img)
            frames.append(img)

        video_tensor = torch.stack(frames)  # (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
        return video_tensor

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_tensor = self._load_video_frames(sample['video_path'])
        label = sample['label']
        return video_tensor, label
