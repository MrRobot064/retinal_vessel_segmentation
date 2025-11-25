import os
import random
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# 0. 一些超参数 (可以按需修改)
# ---------------------------------------------------------
DRIVE_ROOT = "./data/DRIVE/train"
PATCH_SIZE = 64  # DRIVE 上使用 64x64 patch
GAMMA = 1.2  # gamma 校正系数
PATCHES_PER_IMAGE = 200  # 每张原图随机抽多少个 patch (你可以改成 500 或 1000)

# 预处理结果的输出目录
OUT_DIR = "./preprocessed/DRIVE_patches"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. 路径设置
# ---------------------------------------------------------
IMG_DIR = os.path.join(DRIVE_ROOT, "images")
LBL_DIR = os.path.join(DRIVE_ROOT, "labels")
FOV_DIR = os.path.join(DRIVE_ROOT, "fov")

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

    # 3.1 读取原图、label、FOV
    img_path = os.path.join(IMG_DIR, filename)
    label_file = f"{case_id}_manual1.gif"
    lbl_path = os.path.join(LBL_DIR, label_file)

    # FOV mask 文件名匹配 (和你之前一样)
    fov_candidates = [f for f in os.listdir(FOV_DIR) if f.startswith(case_id)]
    if len(fov_candidates) != 1:
        raise RuntimeError(
            f"FOV not found or ambiguous for case {case_id}: {fov_candidates}"
        )
    fov_path = os.path.join(FOV_DIR, fov_candidates[0])

    # 3.2 预处理：灰度 -> 归一化 -> gamma
    gray_norm = load_gray_image(img_path)  # [0,1]
    gray_gamma = np.power(gray_norm, GAMMA)  # gamma 调整
    label_bin = load_binary_mask(lbl_path)  # {0,1}
    fov_bin = load_binary_mask(fov_path)  # {0,1}

    H, W = gray_gamma.shape
    ph, pw = PATCH_SIZE, PATCH_SIZE

    # 3.3 为这张图抽 PATCHES_PER_IMAGE 个 patch
    patches_this_img = 0
    max_tries = PATCHES_PER_IMAGE * 10  # 防止一直抽不到合适的 patch 死循环

    while patches_this_img < PATCHES_PER_IMAGE and max_tries > 0:
        max_tries -= 1

        y, x = random_patch_coord(H, W, ph, pw)
        img_patch = gray_gamma[y : y + ph, x : x + pw]
        lbl_patch = label_bin[y : y + ph, x : x + pw]
        fov_patch = fov_bin[y : y + ph, x : x + pw]

        # 要求：大部分在 FOV 内，且 label 里有一点血管像素（否则全背景没啥用）
        if fov_patch.mean() > 0.8 and lbl_patch.mean() > 0.01:
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
