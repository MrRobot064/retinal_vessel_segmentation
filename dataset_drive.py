import numpy as np
import torch
from torch.utils.data import Dataset

import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image
import random


class DrivePatchDataset(Dataset):
    """
    从预处理好的 npz 文件中读取 DRIVE 的 patch 数据，供 DataLoader 使用。

    约定 npz 文件里包含：
        - images: [N, H, W]，float32，范围 [0,1]，已做灰度+归一化+gamma
        - labels: [N, H, W]，float32，{0,1} 的二值 mask
        - （可选）其它信息，比如 patch_size、gamma 等，读取时不会强依赖

    参数：
        npz_path : str
            预处理生成的 .npz 文件路径
        transform : callable or None
            用于在线数据增强的函数，签名为 transform(img, lbl) -> (img, lbl)，
            其中 img, lbl 均为 numpy 数组 [H, W]。
        return_index : bool
            如果为 True，则 __getitem__ 额外返回一个 index，便于调试 / 可视化。
    """

    def __init__(self, npz_path, transform=None, return_index=False):
        super().__init__()

        data = np.load(npz_path)

        # 必需字段：images & labels
        self.images = data["images"]  # [N, H, W]
        self.labels = data["labels"]  # [N, H, W]

        assert self.images.shape == self.labels.shape, (
            f"images shape {self.images.shape} and labels shape "
            f"{self.labels.shape} mismatch!"
        )

        self.length = self.images.shape[0]
        self.transform = transform
        self.return_index = return_index

        # 可选字段：如果有就记一下，纯记录，不影响训练
        self.patch_size = data.get("patch_size", None)
        self.gamma = data.get("gamma", None)

        print("========================================")
        print("DrivePatchDataset initialized")
        print("========================================")
        print(f"npz file    : {npz_path}")
        print(f"num patches : {self.length}")
        print(f"image shape : {self.images.shape[1:]}  (H, W)")
        if self.patch_size is not None:
            print(f"patch_size  : {self.patch_size}")
        if self.gamma is not None:
            print(f"gamma       : {self.gamma}")
        print("========================================\n")

    def __len__(self):
        """返回样本总数 N"""
        return self.length

    def __getitem__(self, idx):
        """
        返回第 idx 个样本:
            img: torch.FloatTensor, [1, H, W]
            lbl: torch.FloatTensor, [1, H, W]

        如果 return_index=True，则返回 (img, lbl, idx)
        """
        # 1) 取 numpy patch
        img = self.images[idx]  # [H, W], float32, [0,1]
        lbl = self.labels[idx]  # [H, W], float32, {0,1}

        # 2) 可选的在线数据增强（使用下面定义的 DriveAugment）
        if self.transform is not None:
            img, lbl = self.transform(img, lbl)  # 仍然是 numpy [H, W]

        # 3) 转成 torch.Tensor，并加通道维 [C=1, H, W]
        img = torch.from_numpy(img).float().unsqueeze(0)
        lbl = torch.from_numpy(lbl).float().unsqueeze(0)

        if self.return_index:
            return img, lbl, idx
        else:
            return img, lbl


class DriveAugment:
    """
    对应论文里的 data augmentation：
        - rotation
        - random flipping
        - random noise（这里可以先不加或后面再补）
        - random blur
        - random contrast
        - random sharpening（可选）

    注意：
        - 几何变换（旋转、翻转）要对 img 和 lbl 同时做
        - 模糊、对比度、锐化只对 img 做，lbl 不能改灰度
    """

    def __init__(
        self,
        max_rotation=15,
        contrast_factor=0.3,
        blur_prob=0.3,
        sharpen_prob=0.3,
    ):
        self.max_rotation = max_rotation
        self.contrast = T.ColorJitter(contrast=contrast_factor)
        self.blur = T.GaussianBlur(kernel_size=3)
        self.blur_prob = blur_prob
        self.sharpen_prob = sharpen_prob

    def __call__(self, img_np, lbl_np):
        # img_np, lbl_np: numpy [H, W], img in [0,1], lbl in {0,1}

        # 先把 numpy 转成 PIL，方便用 torchvision 的函数
        img = Image.fromarray((img_np * 255).astype(np.uint8))
        lbl = Image.fromarray((lbl_np * 255).astype(np.uint8))

        # 1) 随机旋转（图像 & 标签同步）
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        img = F.rotate(img, angle)
        lbl = F.rotate(lbl, angle)

        # 2) 随机翻转（图像 & 标签同步）
        if random.random() < 0.5:
            img = F.hflip(img)
            lbl = F.hflip(lbl)
        if random.random() < 0.5:
            img = F.vflip(img)
            lbl = F.vflip(lbl)

        # 3) 只对图像做的增强（blur / contrast / sharpen）
        img = self.contrast(img)  # 对比度

        if random.random() < self.blur_prob:
            img = self.blur(img)  # 模糊

        if random.random() < self.sharpen_prob:
            # 用 adjust_sharpness 做锐化
            img = F.adjust_sharpness(img, sharpness_factor=2.0)

        # 4) 转回 numpy，维持原始格式
        img_out = np.array(img).astype(np.float32) / 255.0
        lbl_out = (np.array(lbl) > 0).astype(np.float32)

        return img_out, lbl_out
