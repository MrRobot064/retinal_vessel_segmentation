import os
import random
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# 0. 一些超参数 (按论文对齐)
# ---------------------------------------------------------
DRIVE_ROOT = "./data/DRIVE/train"

PATCH_SIZE = 64  # 论文：DRIVE 使用 64×64 patch
GAMMA = 1.2  # 论文有 gamma 调整，但没写具体数值，你这里用 1.2 没问题
PATCHES_PER_IMAGE = 1000  # 论文：In the training phase, we randomly sampled 1000 patches from each raw image.

# 预处理结果的输出目录
OUT_DIR = "./preprocessed/DRIVE_patches"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. 路径设置
# ---------------------------------------------------------
IMG_DIR = os.path.join(DRIVE_ROOT, "images")
LBL_DIR = os.path.join(DRIVE_ROOT, "labels")
FOV_DIR = os.path.join(DRIVE_ROOT, "fov")  # FOV 只在 eval 时用，这里可以选择不用

img_list = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".tif")])
if len(img_list) == 0:
    raise RuntimeError("No DRIVE training images found. Check folder paths.")

print(f"Found {len(img_list)} training images.")


# ---------------------------------------------------------
# 2. 一些小工具函数
# ---------------------------------------------------------
def load_gray_image(path):
    """读取为灰度图并返回 float32 numpy 数组，范围 [0,1]"""
    img_gray = Image.open(path).convert("L")
    arr = np.array(img_gray).astype(np.float32) / 255.0
    return arr


def load_binary_mask(path):
    """读取为灰度图并二值化为 {0,1} float32"""
    mask = Image.open(path).convert("L")
    arr = (np.array(mask) > 0).astype(np.float32)
    return arr


def random_patch_coord(h, w, ph, pw):
    """在 [0, h-ph], [0, w-pw] 范围内随机选一个左上角坐标 (y,x)"""
    y = random.randint(0, h - ph)
    x = random.randint(0, w - pw)
    return y, x


# ---------------------------------------------------------
# 3. 主循环：遍历每一张图，预处理 + 抽 patch
# ---------------------------------------------------------
all_patches_img = []
all_patches_lbl = []

for filename in img_list:
    case_id = filename.split("_")[0]  # 例：'21_training.tif' -> '21'
    print(f"\nProcessing case: {case_id} ({filename})")

    # 3.1 读取原图、label；FOV 这里暂时只加载，不参与采样条件
    img_path = os.path.join(IMG_DIR, filename)
    label_file = f"{case_id}_manual1.gif"
    lbl_path = os.path.join(LBL_DIR, label_file)

    # 如果你后面 eval 要用 FOV，这里顺便读出来；如果仅训练阶段可以不读取
    fov_candidates = [f for f in os.listdir(FOV_DIR) if f.startswith(case_id)]
    if len(fov_candidates) != 1:
        raise RuntimeError(
            f"FOV not found or ambiguous for case {case_id}: {fov_candidates}"
        )
    fov_path = os.path.join(FOV_DIR, fov_candidates[0])

    # 3.2 预处理：灰度 -> 归一化 -> gamma  （完全对应论文的 pre-processing）
    gray_norm = load_gray_image(img_path)  # [0,1]
    gray_gamma = np.power(gray_norm, GAMMA)  # gamma 调整
    label_bin = load_binary_mask(lbl_path)  # {0,1}
    fov_bin = load_binary_mask(fov_path)  # {0,1}

    H, W = gray_gamma.shape
    ph, pw = PATCH_SIZE, PATCH_SIZE

    # （可选）把 FOV 外的区域直接置 0，避免采到全黑背景
    # 论文没写这一点，你做一点小优化也没问题
    gray_gamma = gray_gamma * fov_bin

    # 3.3 为这张图“随机抽取 1000 个 patch”
    #     对应论文原文：we randomly sampled 1000 patches from each raw image.
    patches_this_img = 0
    max_tries = PATCHES_PER_IMAGE * 5  # 防止极端情况下死循环

    while patches_this_img < PATCHES_PER_IMAGE and max_tries > 0:
        max_tries -= 1

        y, x = random_patch_coord(H, W, ph, pw)
        img_patch = gray_gamma[y : y + ph, x : x + pw]
        lbl_patch = label_bin[y : y + ph, x : x + pw]
        fov_patch = fov_bin[y : y + ph, x : x + pw]

        # 变化 1：不再强制要求“血管比例 > 1%”
        # 变化 2：不再要求 FOV > 0.8，只要 patch 至少有一点在 FOV 内即可
        #    这样更符合“完全随机采样”，同时避免采到完全在 FOV 外的全黑块。
        if fov_patch.mean() > 0.0:  # 至少有部分区域在视野内
            all_patches_img.append(img_patch)
            all_patches_lbl.append(lbl_patch)
            patches_this_img += 1

    print(f"  Collected {patches_this_img} patches for this case.")

# ---------------------------------------------------------
# 4. 把所有 patch 保存到一个 .npz 文件中
# ---------------------------------------------------------
all_patches_img = np.array(all_patches_img, dtype=np.float32)  # [N, H, W]
all_patches_lbl = np.array(all_patches_lbl, dtype=np.float32)  # [N, H, W]

out_path = os.path.join(OUT_DIR, "drive_train_patches_gamma.npz")
np.savez_compressed(
    out_path,
    images=all_patches_img,
    labels=all_patches_lbl,
    patch_size=PATCH_SIZE,
    gamma=GAMMA,
)

print("\nDone!")
print(f"Total patches: {all_patches_img.shape[0]}")
print(f"Saved to: {out_path}")
