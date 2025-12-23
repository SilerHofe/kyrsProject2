import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.unet_segmentation import UNet
from src.text_encoder import TextEncoder
from src.stylegan2_generator import Generator
from src.encoder import AutoencoderGenerator
from natasha import (
    Doc,
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger
)

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


segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
# --------------------
# НАСТРОЙКИ
# --------------------
IMAGE_PATH = "demo_images/1.jpg"
TEXT = "Сделай лицо с черными волосами и глазами"
OUTPUT_PATH = "result.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10
# --------------------
# ЗАГРУЗКА МОДЕЛЕЙ
# --------------------
seg = UNet(num_classes=NUM_CLASSES).to(device)
txt = TextEncoder(device).to(device)
gen = Generator(num_classes=NUM_CLASSES).to(device)

seg.load_state_dict(torch.load(
    "checkpoints/segmentation.pth",
    map_location=device,
    weights_only=True
))
gen.load_state_dict(torch.load(
    "checkpoints/generator.pth",
    map_location=device,
    weights_only=True
))

seg.eval()
txt.eval()
gen.eval()


attrs = parse_text_ru(TEXT)
prompt = attrs_to_prompt(attrs)

print("CLIP PROMPT:", prompt)


# --------------------
# ПРЕОБРАЗОВАНИЕ ИЗОБРАЖЕНИЯ
# --------------------
seg_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# gen_tf = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])

image = Image.open(IMAGE_PATH).convert("RGB")
image_seg = seg_tf(image).unsqueeze(0).to(device)
# image_gen = gen_tf(image).unsqueeze(0).to(device)

# --------------------
# ПРОГОН ПО СИСТЕМЕ
# --------------------
with torch.no_grad():
    # 1. Сегментация
    mask_logits = seg(image_seg)
    mask_idx = mask_logits.argmax(1)
    # mask_idx = torch.nn.functional.interpolate(
    #     mask_idx.unsqueeze(1).float(),
    #     size=(128, 128),
    #     mode="nearest"
    # ).long().squeeze(1)

    mask_oh = torch.nn.functional.one_hot(
        mask_idx, num_classes=NUM_CLASSES
    ).permute(0, 3, 1, 2).float()
    # 2. Текст
    text_vec = txt([prompt]).to(device)
    # 3. Генерация
    output = gen(mask = mask_oh, text_vec = text_vec)

# --------------------
# СОХРАНЕНИЕ РЕЗУЛЬТАТА
# --------------------

output = output.squeeze(0).cpu()
output = (output + 1) / 2
output = transforms.ToPILImage()(output)
output.save(OUTPUT_PATH)


print("Готово! Результат сохранён в", OUTPUT_PATH)