import os
import subprocess

methods = ["threshold", "kmeans", "pix2pix", "spade", "proposed"]

for method in methods:
    cmd = [
        "python", "-m", "pytorch_fid",
        "data/real",
        f"results/{method}"
    ]
    print(f"\nFID for {method}:")
    subprocess.run(cmd)
