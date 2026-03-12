import torch.nn as nn
from architecture import FirstBlock,ResidualBlock,FinalBlock


class PixelCNN(nn.Module):
    """
    PixelCNN model for autoregressive generative modeling of images.

    Architecture:
        1. FirstBlock          — Type-A MaskedConv + BN
        2. n_block × ResidualBlock — Type-B MaskedConv with skip connections
        3. FinalBlock          — projection to logits

    Args:
        n_channel (int): Input channels.
        h (int): Bottleneck dimension.
        discrete_channel (int): Number of discrete values per channel (default 256).
        n_block (int): Number of residual blocks.
    """

    def __init__(self, n_channel: int = 3, h: int = 128,
                 discrete_channel: int = 256, n_block: int = 15):
        super().__init__()
        self.discrete_channel = discrete_channel
        self.n_block = n_block

        self.first_block = FirstBlock(n_channel, 2 * h, k_size=7, stride=1, pad=3)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(h, k_size=3, stride=1, pad=1) for _ in range(n_block)]
        )
        self.final_block = FinalBlock(n_channel, h, discrete_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, c_in, height, width = x.size()

        x = self.first_block(x)       # [B, 2h, H, W]

        for block in self.res_blocks: # [B, 2h, H, W]
            x = block(x)

        x = self.final_block(x)       # [B, C*256, H, W]

        x = x.view(batch_size, c_in,
                   self.discrete_channel,
                   height, width)     # [B, C, 256, H, W]

        x = x.permute(0, 1, 3, 4, 2) # [B, C, H, W, 256]

        return x