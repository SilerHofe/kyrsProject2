from torch.utils.data import DataLoader
import torchvision
import torch
from text_encoder import TextEncoder
from natasha import (
    Doc,
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger
)

from stylegan2_generator import Generator
from dataset_gen import GeneratorDataset

dataset = GeneratorDataset(
    "data/CelebA-HQ-img",
    "data/CelebAMask-HQ-mask-anno"
)

# -------- TEXT PIPELINE --------
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

def parse_text_ru(text):

    """
    Преобразует русский текст в семантические атрибуты
    """
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    attrs = {
        "hair_color": None,
        "has_eyes": False,
        "has_lips": False
    }

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

        if token.lemma in ["волосы", "прическа"]:
            attrs["hair_color"] = "black"

        if token.lemma in ["глаза"]:
            attrs["has_eyes"] = True

        if token.lemma in ["губы", "рот"]:
            attrs["has_lips"] = True

    return attrs

def attrs_to_prompt(attrs):
    """
    Преобразует атрибуты в английский prompt для CLIP
    """
    parts = ["a human face"]

    if attrs["hair_color"]:
        parts.append(f"with {attrs['hair_color']} hair")

    if attrs["has_eyes"]:
        parts.append("with visible eyes")

    if attrs["has_lips"]:
        parts.append("with lips")

    return " ".join(parts)
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = Generator().to(device)
    criterion = torch.nn.L1Loss()

    loader = DataLoader(
        dataset,
        batch_size=32,        # можно 32
        shuffle=True,
        num_workers=8,        # НЕ 8, оптимальнее
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    optimizer = torch.optim.Adam(gen.parameters(), lr=2e-4)


    text_encoder = TextEncoder(device).to(device)
    text_encoder.eval()

    for epoch in range(101):
        for mask, target_img in loader:
            mask = mask.to(device)
            target_img = target_img.to(device)

            B = mask.size(0)
            
            ru_texts = [
                "Сделай лицо с волосами и глазами"
            ] * B
            
            attrs = [parse_text_ru(t) for t in ru_texts]
            prompts = [attrs_to_prompt(a) for a in attrs]

            text_vec = text_encoder(prompts).to(device)

            output = gen(mask, text_vec)

            loss = criterion(output, target_img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            torchvision.utils.save_image(
            output,
            f"debug/gen_epoch{epoch}_step{epoch}.png",
            normalize=True
            )

        print(f"Epoch {epoch}: loss={loss.item():.4f}")
        torch.save(gen.state_dict(), "checkpoints/generator.pth")

if __name__ =="__main__":
    main()