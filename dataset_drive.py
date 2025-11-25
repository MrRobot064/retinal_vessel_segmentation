import numpy as np
import torch
from torch.utils.data import Dataset


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
            其中 img, lbl 均为 numpy 数组 [H, W]，你可以后续自己实现。
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
        img = self.images[idx]  # [H, W], float32
        lbl = self.labels[idx]  # [H, W], float32

        # 2) 可选的在线数据增强（后面可以自己加）
        if self.transform is not None:
            img, lbl = self.transform(img, lbl)

        # 3) 转成 torch.Tensor，并加通道维 [C=1, H, W]
        img = torch.from_numpy(img).float().unsqueeze(0)
        lbl = torch.from_numpy(lbl).float().unsqueeze(0)

        if self.return_index:
            return img, lbl, idx
        else:
            return img, lbl
