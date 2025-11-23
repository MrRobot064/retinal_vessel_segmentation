import os
import glob
import shutil

# ====== 1. 原始 DRIVE 路径（你现在的 raw 结构） ======
RAW_ROOT = "./data/DRIVE/raw"

TRAIN_IMG_DIR = os.path.join(RAW_ROOT, "training", "images")
TRAIN_GT_DIR = os.path.join(RAW_ROOT, "training", "1st_manual")
TRAIN_FOV_DIR = os.path.join(RAW_ROOT, "training", "mask")

TEST_IMG_DIR = os.path.join(RAW_ROOT, "test", "images")
TEST_GT_DIR = os.path.join(RAW_ROOT, "test", "1st_manual")
TEST_FOV_DIR = os.path.join(RAW_ROOT, "test", "mask")

# ====== 2. 输出路径（真正训练/测试用的数据） ======
OUT_ROOT = "./data/DRIVE"

TRAIN_OUT_IMG = os.path.join(OUT_ROOT, "train", "images")
TRAIN_OUT_LABEL = os.path.join(OUT_ROOT, "train", "labels")
TRAIN_OUT_FOV = os.path.join(OUT_ROOT, "train", "fov")

TEST_OUT_IMG = os.path.join(OUT_ROOT, "test", "images")
TEST_OUT_LABEL = os.path.join(OUT_ROOT, "test", "labels")
TEST_OUT_FOV = os.path.join(OUT_ROOT, "test", "fov")

for d in [
    TRAIN_OUT_IMG,
    TRAIN_OUT_LABEL,
    TRAIN_OUT_FOV,
    TEST_OUT_IMG,
    TEST_OUT_LABEL,
    TEST_OUT_FOV,
]:
    os.makedirs(d, exist_ok=True)


def copy_triplet(
    img_path, gt_dir, fov_dir, dst_img_dir, dst_label_dir, dst_fov_dir, split="training"
):
    """
    把一张 fundus 图像 + 对应 vessel label + FOV mask
    复制到目标 images / labels / fov 目录。
    split: "training" 或 "test"，用来确定 mask 的命名规则。
    """
    filename = os.path.basename(img_path)  # 21_training.tif / 01_test.tif
    base, ext = os.path.splitext(filename)  # 21_training / 01_test
    case_id = base.split("_")[0]  # "21" or "01"

    # ---- 1) vessel label (manual1) ----
    gt_name = f"{case_id}_manual1.gif"
    gt_src = os.path.join(gt_dir, gt_name)
    if not os.path.exists(gt_src):
        raise FileNotFoundError(f"GT not found: {gt_src}")

    # ---- 2) FOV mask ----
    if split == "training":
        # e.g. 21_training_mask.gif
        fov_name = f"{case_id}_training_mask.gif"
    else:  # "test"
        # e.g. 01_test_mask.gif
        fov_name = f"{case_id}_test_mask.gif"

    fov_src = os.path.join(fov_dir, fov_name)
    if not os.path.exists(fov_src):
        raise FileNotFoundError(f"FOV mask not found: {fov_src}")

    # ---- 3) 复制到输出目录 ----
    shutil.copy(img_path, os.path.join(dst_img_dir, filename))
    shutil.copy(gt_src, os.path.join(dst_label_dir, gt_name))
    shutil.copy(fov_src, os.path.join(dst_fov_dir, fov_name))


# ====== 3. 处理训练集（官方 20 张） ======
train_imgs = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.tif")))
print("Found training images:", len(train_imgs))

for img in train_imgs:
    copy_triplet(
        img,
        TRAIN_GT_DIR,
        TRAIN_FOV_DIR,
        TRAIN_OUT_IMG,
        TRAIN_OUT_LABEL,
        TRAIN_OUT_FOV,
        split="training",
    )

# ====== 4. 处理测试集（官方 20 张） ======
test_imgs = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.tif")))
print("Found test images:", len(test_imgs))

for img in test_imgs:
    copy_triplet(
        img,
        TEST_GT_DIR,
        TEST_FOV_DIR,
        TEST_OUT_IMG,
        TEST_OUT_LABEL,
        TEST_OUT_FOV,
        split="test",
    )

print("Done! DRIVE reorganized into ./data/DRIVE/train and ./data/DRIVE/test")
