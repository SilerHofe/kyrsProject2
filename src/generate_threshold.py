# generate_threshold.py
import cv2
import os

os.makedirs("results/threshold", exist_ok=True)

for img_name in os.listdir("data/real"):
    img = cv2.imread(f"data/real/{img_name}", 0)
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"results/threshold/{img_name}", mask)
