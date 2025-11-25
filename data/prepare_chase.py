import os
import shutil
import glob

# === 源数据（原始 CHASE 数据）===
RAW_ROOT = "./CHASE/raw"
IMG_DIR = os.path.join(RAW_ROOT, "images")
MASK_DIR = os.path.join(RAW_ROOT, "masks")

# === 输出数据（用于训练/测试）===
OUT_ROOT = "./CHASE"
TRAIN_IMG_DIR = os.path.join(OUT_ROOT, "train", "images")
TRAIN_MASK_DIR = os.path.join(OUT_ROOT, "train", "labels")
TEST_IMG_DIR = os.path.join(OUT_ROOT, "test", "images")
TEST_MASK_DIR = os.path.join(OUT_ROOT, "test", "labels")

# 创建输出目录
for d in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR]:
    os.makedirs(d, exist_ok=True)

# 读取所有图像（按文件名排序）
img_list = sorted(
    glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    + glob.glob(os.path.join(IMG_DIR, "*.png"))
)

print("Found images:", len(img_list))  # 应该是 28 张（CHASE DB1）

# === Zhang & Chung 论文分法 ===
train_imgs = img_list[:20]  # 前 20 张 → train
test_imgs = img_list[20:]  # 后 8 张 → test


def copy_pair(img_path, dst_img_folder, dst_mask_folder):
    """复制 image + mask 到目标文件夹"""
    filename = os.path.basename(img_path)
    stem, ext = os.path.splitext(filename)

    # mask 有两种可能：Image_01_1stHO.png 或 Image_01.png
    mask_candidates = [
        os.path.join(MASK_DIR, stem + "_1stHO.png"),
        os.path.join(MASK_DIR, stem + ".png"),
        os.path.join(MASK_DIR, stem + "_1stHO.jpg"),
        os.path.join(MASK_DIR, stem + ".jpg"),
    ]

    mask_src = None
    for m in mask_candidates:
        if os.path.exists(m):
            mask_src = m
            break

    if mask_src is None:
        raise FileNotFoundError(f"Mask not found for {filename}")

    shutil.copy(img_path, os.path.join(dst_img_folder, filename))
    shutil.copy(mask_src, os.path.join(dst_mask_folder, os.path.basename(mask_src)))


# 复制训练
for img in train_imgs:
    copy_pair(img, TRAIN_IMG_DIR, TRAIN_MASK_DIR)

# 复制测试
for img in test_imgs:
    copy_pair(img, TEST_IMG_DIR, TEST_MASK_DIR)

print("Done! CHASE split saved to ./CHASE")
print(f"Train images: {len(train_imgs)}")
print(f"Test images : {len(test_imgs)}")
