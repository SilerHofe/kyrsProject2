# generate_kmeans.py
import cv2, os, numpy as np

os.makedirs("results/kmeans", exist_ok=True)

for img_name in os.listdir("data/real"):
    img = cv2.imread(f"data/real/{img_name}")
    Z = img.reshape((-1,3)).astype(np.float32)

    _, labels, centers = cv2.kmeans(
        Z, 5, None,
        (cv2.TERM_CRITERIA_EPS, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )

    res = centers[labels.flatten()].reshape(img.shape)
    cv2.imwrite(f"results/kmeans/{img_name}", res)
