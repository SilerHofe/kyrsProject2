from torch.utils.data import DataLoader
import torchvision
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from unet_segmentation import UNet
from dataset import CelebAMaskHQDataset

dataset = CelebAMaskHQDataset(
    "data/CelebA-HQ-img",
    "data/CelebAMask-HQ-mask-anno"
)

PALETTE = {
    0: (0, 0, 0),          # background — чёрный

    1: (90, 90, 90),       # cloth — тёмно-серый
    2: (130, 100, 90),     # neck — тёплый коричневатый
    3: (255, 224, 189),    # skin — телесный
    4: (60, 40, 20),       # hair — тёмно-коричневый

    5: (30, 144, 255),     # eyes / ears / hat — синий
    6: (255, 215, 0),      # brows — золотистый
    7: (220, 20, 60),      # nose — красно-розовый
    8: (0, 200, 0),        # mouth / glasses — зелёный
    9: (180, 60, 180),     # lips — фиолетово-розовый
}

def colorize_mask(mask):
    """
    mask: [H, W] LongTensor или numpy array
    """
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()

    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in PALETTE.items():
        out[mask == cls] = color

    return Image.fromarray(out)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(
        dataset,
        batch_size=32,        # можно 32
        shuffle=True,
        num_workers=6,        # НЕ 8, оптимальнее
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    model = UNet(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(101):
        for img, mask in loader:
            img = img.to(device)
            mask = mask.to(device)

            logits = model(img)
            loss = criterion(logits, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # берём ПЕРВЫЙ элемент последнего батча
                pred = model(img)
                pred_mask = pred.argmax(1)[0].cpu()
                gt_mask = mask[0].cpu()
                img_vis = img[0].cpu()

            # 3️⃣ предсказанная маска
            colorize_mask(pred_mask).save(
                f"debug/seg_pred_epoch{epoch}.png"
            )

            model.train()

        print(f"Epoch {epoch} | Loss {loss.item():.4f}")
        torch.save(model.state_dict(), "checkpoints/segmentation.pth")

if __name__ =="__main__":
    main()