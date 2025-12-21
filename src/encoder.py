import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
    
class RefineBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1)
        )

    def forward(self, x):
        return torch.tanh(x + 0.1 * self.net(x))

class AutoencoderGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.refine = RefineBlock()

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        out = self.refine(x_rec)
        return x_rec, out