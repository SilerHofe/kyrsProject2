import matplotlib.pyplot as plt

methods = ["Threshold", "K-means", "Pix2Pix", "SPADE", "Proposed"]

fid = [89.4, 77.5, 54.1, 32.5, 26.8]
lpips = [0.62, 0.55, 0.43, 0.31, 0.27]

plt.figure()
plt.bar(methods, fid)
plt.title("FID comparison")
plt.ylabel("FID (lower is better)")
plt.show()

plt.figure()
plt.bar(methods, lpips)
plt.title("LPIPS comparison")
plt.ylabel("LPIPS (lower is better)")
plt.show()
