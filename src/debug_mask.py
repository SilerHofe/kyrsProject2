import os
import re
from collections import defaultdict

MASK_ROOT = "data/CelebAMask-HQ-mask-anno"
IMAGE_ID = 0  # поменяй при необходимости

found = defaultdict(int)

for chunk in os.listdir(MASK_ROOT):
    chunk_dir = os.path.join(MASK_ROOT, chunk)
    if not os.path.isdir(chunk_dir):
        continue

    for file in os.listdir(chunk_dir):
        ids = re.findall(r"\d+", file)
        if not ids:
            continue

        if int(ids[0]) == IMAGE_ID:
            for part in [
                "skin", "hair", "eye", "brow",
                "nose", "mouth", "lip", "ear"
            ]:
                if part in file:
                    found[part] += 1

print("НАЙДЕННЫЕ КЛАССЫ:")
for k, v in found.items():
    print(k, v)
