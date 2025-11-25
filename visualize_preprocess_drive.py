import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------
# 配置
# ---------------------------------------------------------
DRIVE_ROOT = "./data/DRIVE/train"
GAMMA = 1.2  # 控制 gamma 校正强度
SAVE_FIG = False  # 如需保存图改为 True
SAVE_PATH = "./preprocessed_visualization.png"  # 保存图片

# ---------------------------------------------------------
# 读取路径
# ---------------------------------------------------------
IMG_DIR = os.path.join(DRIVE_ROOT, "images")
LBL_DIR = os.path.join(DRIVE_ROOT, "labels")
FOV_DIR = os.path.join(DRIVE_ROOT, "fov")

img_list = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".tif")])
if len(img_list) == 0:
    raise RuntimeError("No DRIVE training images found under " + IMG_DIR)

filename = random.choice(img_list)
case_id = filename.split("_")[0]

print("Selected image:", filename)

# ---------------------------------------------------------
# 读取图像（PIL）
# ---------------------------------------------------------
img_rgb_pil = Image.open(os.path.join(IMG_DIR, filename)).convert("RGB")
img_gray_pil = Image.open(os.path.join(IMG_DIR, filename)).convert("L")
lbl_pil = Image.open(os.path.join(LBL_DIR, f"{case_id}_manual1.gif")).convert("L")

# FOV
fov_candidates = [f for f in os.listdir(FOV_DIR) if f.startswith(case_id)]
if len(fov_candidates) != 1:
    raise RuntimeError(
        f"FOV not found or ambiguous for case {case_id}: {fov_candidates}"
    )
fov_pil = Image.open(os.path.join(FOV_DIR, fov_candidates[0])).convert("L")

# ---------------------------------------------------------
# 转成 numpy 并做预处理
# ---------------------------------------------------------
img_rgb = np.asarray(img_rgb_pil)  # [H, W, 3]
gray_np = np.asarray(img_gray_pil).astype(np.float32)  # [H, W]
gray_norm = gray_np / 255.0
gray_gamma = np.power(gray_norm, GAMMA)

label_bin = (np.asarray(lbl_pil) > 0).astype(np.float32)
fov_bin = (np.asarray(fov_pil) > 0).astype(np.float32)

# overlay: 在 gamma 图上叠加血管
overlay = np.dstack([gray_gamma, gray_gamma, gray_gamma])
overlay[label_bin == 1] = [1, 0, 0]

# ---------------------------------------------------------
# 绘图
# ---------------------------------------------------------
plt.figure(figsize=(14, 8))

titles = [
    "Original RGB",
    "Grayscale",
    f"Gamma Adjusted (γ={GAMMA})",
    "Label (manual1)",
    "FOV Mask",
    "Overlay (Gamma + Vessel)",
]

images = [img_rgb, gray_np, gray_gamma, label_bin, fov_bin, overlay]

for i, (title, image) in enumerate(zip(titles, images), 1):
    plt.subplot(2, 3, i)

    img_arr = np.asarray(image)

    if img_arr.ndim == 2:
        plt.imshow(img_arr, cmap="gray")
    else:
        plt.imshow(img_arr)

    plt.title(title, fontsize=14)
    plt.axis("off")

plt.tight_layout()

# 保存高清图
if SAVE_FIG:
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {SAVE_PATH}")

plt.show()
print("Visualization complete.")
