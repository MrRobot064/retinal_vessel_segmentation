import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# ---------------------------------------------------------
# 1. 选择 DRIVE 的 train 文件夹
# ---------------------------------------------------------
DRIVE_ROOT = "./data/DRIVE/train"

IMG_DIR = os.path.join(DRIVE_ROOT, "images")
LBL_DIR = os.path.join(DRIVE_ROOT, "labels")
FOV_DIR = os.path.join(DRIVE_ROOT, "fov")

# 获取所有训练图像
img_list = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".tif")])

if len(img_list) == 0:
    raise RuntimeError("No DRIVE training images found. Check folder paths.")

# 随机选一张图
filename = random.choice(img_list)
case_id = filename.split("_")[0]  # 例： '21_training.tif' → '21'

print("Selected image:", filename)

# ---------------------------------------------------------
# 2. 读取图像、label、fov
# ---------------------------------------------------------
img = Image.open(os.path.join(IMG_DIR, filename)).convert("RGB")

label_file = f"{case_id}_manual1.gif"
lbl = Image.open(os.path.join(LBL_DIR, label_file)).convert("L")

# FOV mask 文件名不同，需要搜索
fov_candidates = [f for f in os.listdir(FOV_DIR) if f.startswith(case_id)]
if len(fov_candidates) != 1:
    raise RuntimeError(
        f"FOV not found or ambiguous for case {case_id}: {fov_candidates}"
    )

fov = Image.open(os.path.join(FOV_DIR, fov_candidates[0])).convert("L")

# ---------------------------------------------------------
# 3. 基础预处理：green channel + CLAHE + normalization
# ---------------------------------------------------------
img_np = np.array(img)
green = img_np[:, :, 1]  # 提取绿色通道

# CLAHE 处理（可视化展示用，老师 review 后再决定是否固定使用）
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
green_clahe = clahe.apply(green)

# 归一化到 [0,1]
green_norm = green / 255.0
green_clahe_norm = green_clahe / 255.0

# 标签和 FOV 变成二值 0/1
label_bin = (np.array(lbl) > 0).astype(np.float32)
fov_bin = (np.array(fov) > 0).astype(np.float32)

# ---------------------------------------------------------
# 4. 画图展示预处理效果
# ---------------------------------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original RGB")
plt.imshow(img)
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Green Channel")
plt.imshow(green, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Green + CLAHE")
plt.imshow(green_clahe, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Label (1st_manual)")
plt.imshow(label_bin, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("FOV Mask")
plt.imshow(fov_bin, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("Overlay: CLAHE + Label")
overlay = green_clahe_norm.copy()
overlay = np.dstack([overlay, overlay, overlay])
overlay[label_bin == 1] = [1, 0, 0]  # 血管用红色标记
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()

print("Demo complete.")
