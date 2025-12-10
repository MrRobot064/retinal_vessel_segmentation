# Retinal Vessel Segmentation – DRIVE Patch-based Pipeline

本项目实现了一个完整的 **DRIVE 视网膜血管分割训练流程**，包含：

* DRIVE 数据预处理（灰度、归一化、Gamma、Patch 抽取）
* Patch 数据校验与可视化
* PyTorch 数据集加载与数据增强
* 小型 U-Net 训练流程（可调试）
* 后续可扩展的滑窗预测接口

---

## 1. 项目结构

```
.
├── preprocess_drive_patches.py        # STEP 1: DRIVE 预处理 & patch 生成
├── visualize_preprocess_drive.py      # STEP 1.5: 单图预处理可视化
├── inspect_drive_patches.py           # STEP 2: patch 分布检查
├── dataset_drive.py                   # STEP 3: Dataset + 数据增强模块
├── train_drive_unet_debug.py          # STEP 4: 训练脚本（U-Net）
├── preprocessed/                      # 自动生成：保存 .npz patch 文件
└── data/
    └── DRIVE/
        ├── train/
        │   ├── images/
        │   ├── labels/
        │   └── fov/
        └── test/
            ├── images/
            ├── labels/
            └── fov/
```

---

## 2. 环境要求

建议使用：

* Python ≥ 3.8
* PyTorch ≥ 1.12
* torchvision
* numpy
* pillow
* matplotlib

安装依赖：

```bash
pip install torch torchvision numpy pillow matplotlib
```

---

## 3. 数据准备：DRIVE 目录结构

确保 DRIVE 数据集放在如下结构：

```
data/DRIVE/train/images/*.tif
data/DRIVE/train/labels/*_manual1.gif
data/DRIVE/train/fov/*.gif

data/DRIVE/test/images/
data/DRIVE/test/labels/
data/DRIVE/test/fov/
```

---

## 4. STEP 1：预处理 + Patch 生成

运行：

```bash
python preprocess_drive_patches.py
```

脚本会执行：

1. 灰度化（convert L）
2. 归一化到 [0,1]
3. Gamma 校正（默认 γ=1.2）
4. FOV 过滤
5. 每张训练图随机采样 1000 个 64×64 patch

输出：

```
preprocessed/DRIVE_patches/drive_train_patches_gamma.npz
```

文件中包含：

* images: (N, 64, 64)
* labels: (N, 64, 64)
* patch_size, gamma 等附加信息

---

## 5. STEP 1.5：单图预处理可视化（可选）

查看 gamma / FOV / label 是否对齐：

```bash
python visualize_preprocess_drive.py
```

输出一张 debug 图，包含：

* 原图 RGB
* 灰度图
* gamma 后图
* label
* FOV
* gamma + vessel overlay

---

## 6. STEP 2：PATCH 分布检查

推荐在训练前检查 patch 是否采样正常：

```bash
python inspect_drive_patches.py
```

脚本会显示：

* 随机 patch 可视化
* 血管像素比例直方图
* patch 统计摘要（纯背景比例等）

可用于确保预处理正确。

---

## 7. STEP 3：Dataset + 数据增强

核心类在：

```
dataset_drive.py
```

使用方法：

```python
from dataset_drive import DrivePatchDataset, DriveAugment

dataset = DrivePatchDataset(
    "./preprocessed/DRIVE_patches/drive_train_patches_gamma.npz",
    transform=DriveAugment(),   # 训练增强
)
```

增强包括：

* 随机旋转
* 随机翻转
* 随机模糊
* 随机关联几何变换
* 对比度调整
* 锐化

（标签严格保持二值，不做模糊/对比度。）

---

## 8. STEP 4：训练 U-Net（可调试）

运行：

```bash
python train_drive_unet_debug.py
```

脚本特点：

* 简易 U-Net（可改成更多 block）
* BCE + Dice loss
* train/val 自动划分
* debug 可视化：每轮保存 overlay（GT=绿，Pred=红）
* 自动保存最优模型：`best_model.pth`
