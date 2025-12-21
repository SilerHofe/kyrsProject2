import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_classes=10, text_dim=512):
        super().__init__()

        self.text_fc = nn.Linear(text_dim, 128)

        self.net = nn.Sequential(
            nn.Conv2d(num_classes + 128, 256, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 3, 1),
            nn.Tanh()
        )

    def forward(self, mask, text_vec):
        B, _, H, W = mask.shape

        t = self.text_fc(text_vec)
        t = t[:, :, None, None].repeat(1, 1, H, W)

        x = torch.cat([mask, t], dim=1)
        residual = self.net(x)

        return residual
