import torch
import clip

class TextEncoder(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.model.eval()

        # замораживаем веса
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, texts):
        """
        texts: list[str]
        return: Tensor [B, 512]
        """
        tokens = clip.tokenize(texts).to(next(self.model.parameters()).device)

        with torch.no_grad():
            text_features = self.model.encode_text(tokens)

        return text_features.float()
