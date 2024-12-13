#discriminator
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # (3, 256, 256) -> (64, 128, 128)
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (64, 128, 128) -> (128, 64, 64)
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (128, 64, 64) -> (256, 32, 32)
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (256, 32, 32) -> (512, 16, 16)
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (512, 16, 16) -> (1, 16, 16)
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)