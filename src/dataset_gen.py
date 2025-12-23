import os
import numpy as np
import torch
import re
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# ⚠️ должно совпадать с генератором
NUM_CLASSES = 10

CLASS_PRIORITY = [
    ("cloth", 1),
    ("neck", 2),
    ("skin", 3),
    ("hair", 4),

    ("l_eye", 5),
    ("r_eye", 5),
    ("r_ear", 5),
    ("l_ear", 5),
    ("hat", 5),

    ("l_brow", 6),
    ("r_brow", 6),

    ("nose", 7),
    
    ("eye_g", 8),
    ("mouth", 8),
    
    ("u_lip", 9),
    ("l_lip", 9),
]

class GeneratorDataset(Dataset):
    def __init__(self, image_dir, mask_root, max_samples=16):
        self.image_dir = image_dir
        self.mask_root = mask_root
        self.images = sorted(os.listdir(image_dir))[:max_samples]

        # генератор работает на 128
        self.img_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.mask_tf = transforms.Resize(
            (256, 256),
            interpolation=Image.NEAREST
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img_id = int(re.findall(r"\d+", name)[0])

        # изображение
        img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        img = self.img_tf(img)

        # собираем mask_idx
        mask = np.zeros((256, 256), dtype=np.uint8)

        for chunk in os.listdir(self.mask_root):
            chunk_dir = os.path.join(self.mask_root, chunk)
            if not os.path.isdir(chunk_dir):
                continue

            for file in os.listdir(chunk_dir):
                ids = re.findall(r"\d+", file)
                if not ids or int(ids[0]) != img_id:
                    continue

                for key, cls in CLASS_PRIORITY:
                    if key in file:
                        m = Image.open(os.path.join(chunk_dir, file)).convert("L")
                        m = self.mask_tf(m)
                        m = np.array(m) > 0
                        mask[(m) & (mask == 0)] = cls

        # 👉 one-hot маска
        mask = torch.from_numpy(mask).long()
        mask_oh = F.one_hot(mask, num_classes=NUM_CLASSES)
        mask_oh = mask_oh.permute(2, 0, 1).float()

        # 👉 target = исходное изображение
        target = img.clone()

        return mask_oh, target
