import lpips
import torch
import torchvision.transforms as T
from PIL import Image
import os

loss_fn = lpips.LPIPS(net='alex')
transform = T.Compose([
    T.Resize((128,128)),
    T.ToTensor()
])

def load_images(folder):
    imgs = []
    for name in os.listdir(folder):
        img = Image.open(os.path.join(folder, name)).convert("RGB")
        imgs.append(transform(img).unsqueeze(0))
    return imgs

real = load_images("data/real")

methods = ["threshold", "kmeans", "pix2pix", "spade", "proposed"]

for method in methods:
    fake = load_images(f"results/{method}")
    scores = []

    for r, f in zip(real, fake):
        score = loss_fn(r, f)
        scores.append(score.item())

    print(method, sum(scores) / len(scores))
