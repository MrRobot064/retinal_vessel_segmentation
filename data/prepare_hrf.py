import os
import glob
import shutil
import random

# --------- 原始 HRF 根目录 ---------
ROOT = "./HRF/raw"

IMG_DIR = os.path.join(ROOT, "images")
GT_DIR = os.path.join(ROOT, "manual1")
FOV_DIR = os.path.join(ROOT, "mask")

# --------- 输出目标目录（train / test） ---------
OUT_ROOT = "./HRF"

TRAIN_IMG_DIR = os.path.join(OUT_ROOT, "train", "images")
TRAIN_GT_DIR = os.path.join(OUT_ROOT, "train", "labels")
TRAIN_FOV_DIR = os.path.join(OUT_ROOT, "train", "fov")

TEST_IMG_DIR = os.path.join(OUT_ROOT, "test", "images")
TEST_GT_DIR = os.path.join(OUT_ROOT, "test", "labels")
TEST_FOV_DIR = os.path.join(OUT_ROOT, "test", "fov")

# 创建输出目录
for d in [
    TRAIN_IMG_DIR,
    TRAIN_GT_DIR,
    TRAIN_FOV_DIR,
    TEST_IMG_DIR,
    TEST_GT_DIR,
    TEST_FOV_DIR,
]:
    os.makedirs(d, exist_ok=True)

# --------- 扫描原始 fundus 图像 ---------
img_paths = []
for ext in ["*.jpg", "*.JPG", "*.png", "*.tif"]:
    img_paths.extend(glob.glob(os.path.join(IMG_DIR, ext)))

img_paths = sorted(img_paths)
print("Found images:", len(img_paths))

if len(img_paths) == 0:
    raise RuntimeError("没有找到 HRF 图像，请检查 data/HRF/raw/all/images")

# --------- 随机划分：70% 训练, 30% 测试 ---------
random.seed(42)
random.shuffle(img_paths)

split_idx = int(0.7 * len(img_paths))
train_imgs = img_paths[:split_idx]
test_imgs = img_paths[split_idx:]

print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")


def copy_triplet(img_path, dst_img_dir, dst_gt_dir, dst_fov_dir):
    """复制 image + manual1 + FOV mask 到目标目录"""
    filename = os.path.basename(img_path)
    stem, _ = os.path.splitext(filename)

    gt_src = os.path.join(GT_DIR, stem + ".tif")
    fov_src = os.path.join(FOV_DIR, stem + "_mask.tif")

    if not os.path.exists(gt_src):
        raise FileNotFoundError(f"找不到 manual1: {gt_src}")

    if not os.path.exists(fov_src):
        raise FileNotFoundError(f"找不到 FOV mask: {fov_src}")

    shutil.copy(img_path, os.path.join(dst_img_dir, filename))
    shutil.copy(gt_src, os.path.join(dst_gt_dir, stem + ".tif"))
    shutil.copy(fov_src, os.path.join(dst_fov_dir, stem + "_mask.tif"))


# --------- 复制训练集 ---------
for p in train_imgs:
    copy_triplet(p, TRAIN_IMG_DIR, TRAIN_GT_DIR, TRAIN_FOV_DIR)

# --------- 复制测试集 ---------
for p in test_imgs:
    copy_triplet(p, TEST_IMG_DIR, TEST_GT_DIR, TEST_FOV_DIR)

print("HRF dataset prepared successfully!")
print("Output path: ./HRF")
