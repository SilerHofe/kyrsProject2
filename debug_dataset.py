import matplotlib.pyplot as plt
from src.dataset import CelebAMaskHQDataset

ds = CelebAMaskHQDataset(
    "data/CelebA-HQ-img",
    "data/CelebAMask-HQ-mask-anno",
    max_samples=1
)

img, mask = ds[0]

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Image")
plt.imshow((img.permute(1,2,0) * 0.5 + 0.5))
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Mask")
plt.imshow(mask, cmap="tab20")
plt.axis("off")

plt.show()
