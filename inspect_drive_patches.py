import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. 配置区：根据自己情况改这里
# ---------------------------------------------------------

# 预处理脚本生成的 npz 文件路径
NPZ_PATH = "./preprocessed/DRIVE_patches/drive_train_patches_gamma.npz"

# 随机可视化多少个 patch（16 或 25 都比较合适）
NUM_VIS_PATCHES = 16

# 血管比例直方图的 bin 数量
HIST_BINS = 30

# 生成脚本用到的阈值，仅用于打印提醒
PATCHES_PER_IMAGE = 200  # 每张图希望采集多少 patch
GAMMA = 1.2  # 预处理时使用的 gamma 值（用于记录）


# ---------------------------------------------------------
# 1. 加载 npz 数据
# ---------------------------------------------------------

if not os.path.exists(NPZ_PATH):
    raise FileNotFoundError(f"NPZ file not found: {NPZ_PATH}")

data = np.load(NPZ_PATH)
imgs = data["images"]  # [N, H, W], 预处理后的灰度 + gamma
lbls = data["labels"]  # [N, H, W], 对应的血管 label

N, H, W = imgs.shape
print("========================================")
print("Loaded preprocessed patch data")
print("========================================")
print(f"File        : {NPZ_PATH}")
print(f"Num patches : {N}")
print(f"Patch size  : {H} x {W}")
print("----------------------------------------")
print(f"Configured in gen script (for reference):")
print(f"  GAMMA                = {GAMMA}")
print(f"  PATCHES_PER_IMAGE    = {PATCHES_PER_IMAGE}")
print("========================================\n")


# ---------------------------------------------------------
# 2. #1 可视化：随机看 NUM_VIS_PATCHES 个 patch
# ---------------------------------------------------------

num_vis = min(NUM_VIS_PATCHES, N)
idxs = np.random.choice(N, size=num_vis, replace=False)

cols = int(np.sqrt(num_vis))
rows = int(np.ceil(num_vis / cols))

plt.figure(figsize=(3 * cols, 3 * rows))
for i, idx in enumerate(idxs, 1):
    img = imgs[idx]  # [H, W], float32 [0,1]
    lbl = lbls[idx]  # [H, W], {0,1}

    # 构造 overlay：灰度图 + 红色血管
    overlay = np.dstack([img, img, img])  # 3 通道
    overlay[lbl == 1] = [1.0, 0.0, 0.0]

    plt.subplot(rows, cols, i)
    plt.imshow(overlay)
    plt.title(f"Patch #{idx}")
    plt.axis("off")

plt.suptitle(f"Random {num_vis} patches (gray+gamma with vessel overlay)", fontsize=14)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 3. #2 血管比例分布：统计 & 画直方图
# ---------------------------------------------------------

# 计算每个 patch 的血管比例：血管像素数 / 总像素
ratios = lbls.reshape(N, -1).mean(axis=1)

print("========================================")
print("Vessel ratio statistics (per patch)")
print("========================================")
print(f"Mean ratio : {ratios.mean():.6f}")
print(f"Std  ratio : {ratios.std():.6f}")
print(f"Min  ratio : {ratios.min():.6f}")
print(f"1%   quant : {np.quantile(ratios, 0.01):.6f}")
print(f"5%   quant : {np.quantile(ratios, 0.05):.6f}")
print(f"Median     : {np.median(ratios):.6f}")
print(f"95%  quant : {np.quantile(ratios, 0.95):.6f}")
print(f"99%  quant : {np.quantile(ratios, 0.99):.6f}")
print(f"Max  ratio : {ratios.max():.6f}")
print("========================================\n")

plt.figure(figsize=(6, 4))
plt.hist(ratios, bins=HIST_BINS)
plt.xlabel("Vessel pixel ratio in patch")
plt.ylabel("Count")
plt.title("Distribution of vessel ratios per patch")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 4. 微调建议
# ---------------------------------------------------------

print("========================================")
print("Heuristic suggestions")
print("========================================")

mean_ratio = ratios.mean()
low_ratio_frac = (ratios < 0.005).mean()
high_ratio_frac = (ratios > 0.5).mean()

if mean_ratio < 0.01:
    print("- 平均血管比例 < 1%：")
    print("  很多 patch 可能血管太少，模型可能更难学到前景。")
    print("  可以考虑：")
    print("  * 降低生成脚本中的 lbl_patch.mean() 阈值，或")
    print("  * 增加 PATCHES_PER_IMAGE，让采样更丰富。")

if low_ratio_frac > 0.5:
    print(f"- 有 {low_ratio_frac*100:.1f}% 的 patch 血管比例 < 0.5%：")
    print("  说明大量 patch 几乎全是背景，可以适当提高血管像素占比的下限，")
    print("  如把 lbl_patch.mean() > 0.01 改成 > 0.005 或调整采样策略。")

if high_ratio_frac > 0.2:
    print(f"- 有 {high_ratio_frac*100:.1f}% 的 patch 血管比例 > 50%：")
    print("  说明很多 patch 可能在粗主干上，")
    print("  可以考虑多探索细小血管区域（比如改采样策略，更关注细血管区域）。")

if 0.01 <= mean_ratio <= 0.2 and low_ratio_frac < 0.3 and high_ratio_frac < 0.3:
    print("- 血管比例分布看起来比较健康，整体上这些 patch 质量应该是可以用来训练的。")
