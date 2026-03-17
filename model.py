import torch
import torch.nn as nn
from architecture import (FirstBlock, ResidualBlock, FinalBlock,
                          MaskedConv, ResidualRowLSTMBlock,
                          GatedPixelCNNBlock, Encoder)


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

class PixelRNN(nn.Module):
    """PixelRNN model using Row-LSTM residual blocks for autoregressive generation.

    Args:
        n_channel: Number of image channels (1 for MNIST, 3 for CIFAR-10).
        h: Hidden dimension (the masked conv produces 2*h channels).
        n_block: Number of ResidualRowLSTMBlock layers.
        discrete_channel: Number of discrete intensity values (default 256).
    """

    def __init__(self, n_channel=3, h=32, n_block=12, discrete_channel=256):
        super(PixelRNN, self).__init__()

        self.discrete_channel = discrete_channel

        self.MaskAConv = MaskedConv('A', n_channel, 2 * h, k_size=7, stride=1, pad=3)
        block_residual = []

        for i in range(n_block):
          block_residual.append(ResidualRowLSTMBlock(2*h))

        self.block_residual = nn.Sequential(*block_residual)

        # 1x1 conv to 3x256 channels
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2 * h, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, n_channel * discrete_channel, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        """
        Args:
            x: [batch_size, channel, height, width]
        Return:
            out [batch_size, channel, height, width, 256]
        """
        batch_size, c_in, height, width = x.size()

        # [batch_size, 2h, 32, 32]
        x = self.MaskAConv(x)

        # ejecutar bloques residuales RowLSTM
        # [batch_size, 2h, 32, 32]
        x = self.block_residual(x)

        # [batch_size, 3x256, 32, 32]
        x = self.out(x)

        # [batch_size, 3, 256, 32, 32]
        x = x.view(batch_size, c_in, self.discrete_channel, height, width)

        # [batch_size, 3, 32, 32, 256]
        x = x.permute(0, 1, 3, 4, 2)

        return x
    
class GatedPixelCNN(nn.Module):
    """Gated PixelCNN with vertical and horizontal stacks.

    Eliminates the blind spot of the original PixelCNN by separating
    vertical context (rows above) from horizontal context (pixels to
    the left on the current row) and combining them via a learned projection.

    Args:
        in_channels: Number of image channels.
        channels: Number of feature channels throughout the network.
        n_layers: Number of GatedPixelCNNBlock layers.
    """

    def __init__(self, in_channels=3, channels=64, n_layers=12):
        super().__init__()

        self.in_channels = in_channels  # store to reshape output correctly

        self.input_conv = MaskedConv('A', in_channels, channels,
                                     k_size=7, stride=1, pad=3)

        self.blocks = nn.ModuleList(
            [GatedPixelCNNBlock(channels) for _ in range(n_layers)]
        )

        self.out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, in_channels * 256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-channel logits of shape (B, C, H, W, 256)."""
        v = self.input_conv(x)
        h = v.clone()

        for block in self.blocks:
            v, h = block(v, h)

        x = self.out(h)

        B, _, H, W = x.shape

        x = x.view(B, self.in_channels, 256, H, W)
        x = x.permute(0, 1, 3, 4, 2)  # [B, C, H, W, 256]

        return x
    
class ConditionalPixelCNN(PixelCNN):
    """PixelCNN decoder conditioned on a latent vector z.

    Extends PixelCNN by projecting z onto the feature space and adding
    it after the first masked convolution block.

    Args:
        n_channel: Number of image channels.
        h: Bottleneck dimension.
        latent_dim: Dimensionality of the conditioning latent vector.
        discrete_channel: Number of discrete intensity values.
    """

    def __init__(self, n_channel=3, h=32, latent_dim=128, discrete_channel=256):

        super().__init__(n_channel, h, discrete_channel)

        self.z_proj = nn.Linear(latent_dim, 2*h)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return logits conditioned on latent z, shape (B, C, H, W, 256)."""
        batch_size, c_in, height, width = x.size()

        cond = self.z_proj(z).unsqueeze(-1).unsqueeze(-1)

        x = self.first_block(x)

        x = x + cond

        for block in self.res_blocks:
            x = block(x)

        x = self.final_block(x)

        x = x.view(batch_size, c_in, self.discrete_channel, height, width)

        x = x.permute(0,1,3,4,2)

        return x
    
class PixelCNNAutoencoder(nn.Module):
    """Full autoencoder combining a convolutional Encoder with a ConditionalPixelCNN decoder.

    The encoder maps an input image to a latent vector z; the decoder
    reconstructs the image autoregressively conditioned on z.
    """

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()

        self.decoder = ConditionalPixelCNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode x to z, then decode autoregressively conditioned on z."""
        z = self.encoder(x)

        out = self.decoder(x, z)

        return out