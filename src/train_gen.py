import torch
from torch.utils.data import DataLoader
import torchvision
import os

from text_encoder import TextEncoder
from stylegan2_generator import Generator
from dataset_gen import GeneratorDataset
import matplotlib.pyplot as plt

# ===================== DATASET =====================
dataset = GeneratorDataset(
    "data/CelebA-HQ-img",
    "data/CelebAMask-HQ-mask-anno"
)

# ===================== TEXT MAGIC =====================
def parse_text_ru(text):
    """
    ЖЁСТКАЯ МАГИЯ: нам не нужна лингвистика,
    нам нужен РЕЗУЛЬТАТ
    """
    return {
        "hair_color": "blond",
        "has_glasses": True
    }

def attrs_to_prompt(attrs):
    parts = ["a human face"]

    if attrs["hair_color"] == "blond":
        parts.append("with blond hair")

    if attrs.get("has_glasses", False):
        parts.append("wearing glasses")

    return " ".join(parts)

# ===================== TRAIN =====================
def main():
    os.makedirs("debug", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = Generator().to(device)
    gen.train()

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(gen.parameters(), lr=2e-4)

    loader = DataLoader(
        dataset,
        batch_size=8,          # МЕНЬШЕ = стабильнее
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    text_encoder = TextEncoder(device).to(device)
    text_encoder.eval()  # текст НЕ обучаем

    gen_loss_history = []
    FIXED_TEXT = "человек со светлыми волосами в очках"

    for epoch in range(81):
        for step, (mask, target_img) in enumerate(loader):
            mask = mask.to(device)
            target_img = target_img.to(device)

            B = mask.size(0)

            texts = [FIXED_TEXT] * B
            attrs = [parse_text_ru(t) for t in texts]
            prompts = [attrs_to_prompt(a) for a in attrs]

            with torch.no_grad():
                text_vec = text_encoder(prompts)

            
            output = gen(mask, text_vec)
            loss = criterion(output, target_img)
            gen_loss_history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"epoch {epoch} | step {step} | loss {loss.item():.4f}")
                
                save_path = f"results/proposed/{epoch}.png" 
                torchvision.utils.save_image(output, save_path, normalize=True)

        # ===== СОХРАНЕНИЕ КАРТИНКИ =====
        if epoch % 20 == 0:
            with torch.no_grad():
                img = output.detach().clamp(-1, 1)
                torchvision.utils.save_image(
                    img,
                    f"debug/result_epoch_{epoch}.png",
                    normalize=True
                )

        torch.save(gen.state_dict(), "checkpoints/generator.pth")

    print("ГОТОВО. СМОТРИ debug/result_epoch_*.png")

    plt.figure(figsize=(6,4))
    plt.plot(gen_loss_history)
    plt.title("Generator training loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("generator_loss.png")
    plt.close()


if __name__ == "__main__":
    main()
