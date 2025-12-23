import torch
import torch.nn.functional as F
from stylegan2_generator import Generator
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# ====== ЗАГРУЗКА ГЕНЕРАТОРА ======
gen = Generator(num_classes=10, text_dim=512).to(device)
gen.load_state_dict(torch.load("checkpoints/generator.pth", map_location=device))
gen.eval()

for p in gen.parameters():
    p.requires_grad = False

# ====== ФИКТИВНАЯ МАСКА ======
# Можно вообще нули
H = W = 256
mask = torch.zeros(1, 10, H, W, device=device)

# ====== ОПТИМИЗИРУЕМЫЙ ТЕКСТОВЫЙ ВЕКТОР ======
text_vec = torch.randn(1, 512, device=device, requires_grad=True)

optimizer = torch.optim.Adam([text_vec], lr=0.05)

# ====== ОБЛАСТИ ИЗОБРАЖЕНИЯ ======
hair_region = slice(20, 90)      # верх головы
eyes_region = slice(100, 140)    # глаза

# ====== ОПТИМИЗАЦИЯ ======
for step in range(300):
    optimizer.zero_grad()

    img = gen(mask, text_vec)
    img = (img + 1) / 2  # [0,1]

    # Светлые волосы
    hair = img[:, :, hair_region, :]
    hair_loss = -hair.mean()

    # Очки = контраст в области глаз
    eyes = img[:, :, eyes_region, :]
    eyes_contrast = eyes.std()

    loss = hair_loss - 0.5 * eyes_contrast
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"step {step} | loss {loss.item():.4f}")

# ====== СОХРАНЕНИЕ ======
img = img.clamp(0, 1)
img_np = (img[0].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

Image.fromarray(img_np).save("result_blond_glasses.png")
print("ГОТОВО: result_blond_glasses.png")
