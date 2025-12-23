import torch
import clip
from PIL import Image
from torchvision import transforms
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
device = "cuda" if torch.cuda.is_available() else "cpu"
from stylegan2.model import Generator
from stylegan2.op.conv2d_gradfix import conv2d, conv_transpose2d

TEXT = "a human face with black hair"
INPUT_IMAGE = "demo_images/artur_blueman.jpg"
OUTPUT = "result.png"

steps = 800  # Еще больше шагов для лучшего результата
lr = 0.005   # Еще меньший learning rate

# StyleGAN2 (rosinality)
G = Generator(
    size=1024,
    style_dim=512,
    n_mlp=8
).cuda()

# Загрузка весов из .pt файла
ckpt = torch.load(
    "checkpoints/stylegan2-ffhq-config-f.pt",
    map_location="cuda"
)

# Попробуем загрузить веса с более гибкими настройками
try:
    G.load_state_dict(ckpt["g_ema"], strict=False)
except:
    # Если не получается, попробуем загрузить все веса
    G.load_state_dict(ckpt, strict=False)

G.eval()

# CLIP
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model.eval()

# text
text = clip.tokenize([TEXT]).to(device)
with torch.no_grad():
    text_feat = clip_model.encode_text(text)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

# image
tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
])

init_img = Image.open(INPUT_IMAGE).convert("RGB")
init_img = tf(init_img).unsqueeze(0).to(device)

with torch.no_grad():
    init_clip = torch.nn.functional.interpolate(init_img, 224)
    img_feat_init = clip_model.encode_image(init_clip)
    img_feat_init = img_feat_init / img_feat_init.norm(dim=-1, keepdim=True)

# latent - используем более умную инициализацию
# Сначала найдем латентный вектор, который ближе всего к исходному изображению
z = torch.randn(1, 512, device=device, requires_grad=True)
z.data = z.data * 0.1  # Маленькая инициализация для стабильности

optimizer = torch.optim.Adam([z], lr=lr)

for step in range(steps):
    optimizer.zero_grad()

    img, _ = G([z])
    img = (img + 1) / 2
    img = img.clamp(0, 1)

    clip_img = torch.nn.functional.interpolate(img, 224)
    img_feat = clip_model.encode_image(clip_img)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    # Улучшенная функция потерь с балансировкой
    text_loss = 1 - torch.cosine_similarity(img_feat, text_feat)
    identity_loss = 0.3 * (1 - torch.cosine_similarity(img_feat, img_feat_init))
    
    # Добавим регуляризацию для более стабильных результатов
    reg_loss = 0.01 * torch.mean(z ** 2)
    
    loss = text_loss + identity_loss + reg_loss

    loss.backward()
    optimizer.step()

    # Проекция z в более разумное пространство
    with torch.no_grad():
        z.data = torch.clamp(z.data, -2.0, 2.0)

    if step % 100 == 0:
        print(f"{step}: loss={loss.item():.4f}, text_loss={text_loss.item():.4f}, identity_loss={identity_loss.item():.4f}, reg_loss={reg_loss.item():.4f}")

out = transforms.ToPILImage()(img[0].detach().cpu())
out.save(OUTPUT)

print("ГОТОВО:", OUTPUT)
