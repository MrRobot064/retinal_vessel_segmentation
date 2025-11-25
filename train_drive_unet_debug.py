import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from dataset_drive import DrivePatchDataset  # 确保 dataset_drive.py 在同一目录


# ============================================
# 1. 一些超参数
# ============================================
NPZ_PATH = "./preprocessed/DRIVE_patches/drive_train_patches_gamma.npz"

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
VAL_RATIO = 0.2
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "./debug_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================
# 2. 一个小 U-Net 实现（简化版）
# ============================================
class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(
                in_ch // 2, in_ch // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1: from decoder (up), x2: skip from encoder
        x1 = self.up(x1)

        # 对齐尺寸（理论上 64x64 不需要，但保险一点）
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = nn.functional.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        # 拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetSmall(nn.Module):
    """
    一个简化版 U-Net：
    输入: [B, 1, 64, 64]
    输出: [B, 1, 64, 64] (logits)
    """

    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.up1 = Up(128 + 64, 64, bilinear)
        self.up2 = Up(64 + 32, 32, bilinear)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)  # [B,32,64,64]
        x2 = self.down1(x1)  # [B,64,32,32]
        x3 = self.down2(x2)  # [B,128,16,16]
        x = self.up1(x3, x2)  # [B,64,32,32]
        x = self.up2(x, x1)  # [B,32,64,64]
        logits = self.outc(x)  # [B,1,64,64]
        return logits


# ============================================
# 3. 损失函数：BCEWithLogits + Dice
# ============================================
bce_loss_fn = nn.BCEWithLogitsLoss()


def dice_loss(logits, targets, eps=1e-6):
    """
    logits: [B,1,H,W]
    targets: [B,1,H,W]
    """
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1) + eps
    dice = 2 * intersection / union
    return 1 - dice.mean()


def combined_loss(logits, targets, alpha=0.5):
    bce = bce_loss_fn(logits, targets)
    dice = dice_loss(logits, targets)
    return alpha * bce + (1 - alpha) * dice, bce, dice


# ============================================
# 4. 训练 & 验证循环
# ============================================
def train_one_epoch(model, loader, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0

    for step, (imgs, lbls) in enumerate(loader):
        imgs = imgs.to(device)  # [B,1,64,64]
        lbls = lbls.to(device)  # [B,1,64,64]

        optimizer.zero_grad()
        logits = model(imgs)
        loss, bce, dice = combined_loss(logits, lbls)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_bce += bce.item() * imgs.size(0)
        running_dice += dice.item() * imgs.size(0)

        if (step + 1) % 20 == 0:
            print(
                f"  [train] step {step+1}/{len(loader)} "
                f"loss={loss.item():.4f} bce={bce.item():.4f} dice={dice.item():.4f}"
            )

    n = len(loader.dataset)
    return running_loss / n, running_bce / n, running_dice / n


def eval_one_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            logits = model(imgs)
            loss, bce, dice = combined_loss(logits, lbls)

            running_loss += loss.item() * imgs.size(0)
            running_bce += bce.item() * imgs.size(0)
            running_dice += dice.item() * imgs.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_bce / n, running_dice / n


# ============================================
# 5. 可视化：保存一些预测 patch
# ============================================
def save_debug_predictions(model, loader, device, epoch, out_dir, num_samples=8):
    model.eval()
    imgs, lbls = next(iter(loader))  # 从验证集中取一批
    imgs = imgs.to(device)
    lbls = lbls.to(device)

    with torch.no_grad():
        logits = model(imgs)
        probs = torch.sigmoid(logits)

    imgs_np = imgs.cpu().numpy()[:, 0]  # [B,H,W]
    lbls_np = lbls.cpu().numpy()[:, 0]
    preds_np = (probs.cpu().numpy()[:, 0] > 0.5).astype(np.float32)

    num = min(num_samples, imgs_np.shape[0])
    cols = 4
    rows = int(np.ceil(num / cols))

    plt.figure(figsize=(4 * cols, 3 * rows))
    for i in range(num):
        img = imgs_np[i]
        gt = lbls_np[i]
        pd = preds_np[i]

        overlay = np.dstack([img, img, img])
        overlay[gt == 1] = [0, 1, 0]  # 绿: GT
        overlay[pd == 1] = [1, 0, 0]  # 红: Pred

        plt.subplot(rows, cols, i + 1)
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"sample {i}")

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"epoch_{epoch:03d}_preds.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [debug] saved predictions to {save_path}")


# ============================================
# 6. 主函数
# ============================================
def main():
    print("Using device:", DEVICE)

    # 6.1 构建数据集
    dataset = DrivePatchDataset(NPZ_PATH)

    # 简单按 patch 随机划分 train/val
    n_total = len(dataset)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Total patches : {n_total}")
    print(f"Train patches : {n_train}")
    print(f"Val patches   : {n_val}")

    # 6.2 模型 & 优化器
    model = UNetSmall(n_channels=1, n_classes=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(OUT_DIR, "best_unet_drive_debug.pth")

    # 6.3 训练循环
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n====== Epoch {epoch}/{NUM_EPOCHS} ======")
        t0 = time.time()

        train_loss, train_bce, train_dice = train_one_epoch(
            model, train_loader, optimizer, epoch, DEVICE
        )
        val_loss, val_bce, val_dice = eval_one_epoch(model, val_loader, DEVICE)

        dt = time.time() - t0
        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} (bce={train_bce:.4f}, dice={train_dice:.4f}) | "
            f"val_loss={val_loss:.4f} (bce={val_bce:.4f}, dice={val_dice:.4f}) | "
            f"time={dt:.1f}s"
        )

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  [checkpoint] saved best model to {best_ckpt_path}")

        # 每 2 个 epoch 保存一次预测可视化
        if epoch % 2 == 0:
            save_debug_predictions(
                model, val_loader, DEVICE, epoch, OUT_DIR, num_samples=8
            )

    print("\nTraining done.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {best_ckpt_path}")


if __name__ == "__main__":
    main()
