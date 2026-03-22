import torch
import torch.nn as nn
from architecture import (FirstBlock, ResidualBlock, FinalBlock,
                          MaskedConv, ResidualRowLSTMBlock,
                          GatedPixelCNNBlock, Encoder)


class PixelCNN(nn.Module):
    """
    PixelCNN model for autoregressive generative modeling of images.

    Architecture:

        1. FirstBlock              - Type-A MaskedConv + BN
        2. n_block x ResidualBlock - Type-B MaskedConv with skip connections
        3. FinalBlock              - projection to logits

    Args:
        n_channel (int): Number of input image channels.
        h (int): Bottleneck dimension.
        discrete_channel (int): Number of discrete values per channel (default 256).
        n_block (int): Number of residual blocks.
    """

    def __init__(self, n_channel: int = 3, h: int = 128,
                 discrete_channel: int = 256, n_block: int = 15):
        super().__init__()
        self.discrete_channel = discrete_channel
        self.n_block = n_block

        self.first_block = FirstBlock(n_channel, 2 * h, kernel_size=7, stride=1, pad=3)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(h, kernel_size=3, stride=1, pad=1) for _ in range(n_block)]
        )
        self.final_block = FinalBlock(n_channel, h, discrete_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PixelCNN.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (B, C, H, W, 256).
        """
        batch_size, c_in, height, width = x.size()

        x = self.first_block(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.final_block(x)                       
        x = x.reshape(batch_size, c_in,
                       self.discrete_channel,
                       height, width)                  
        x = x.permute(0, 1, 3, 4, 2)                  
        return x


class PixelRNN(nn.Module):
    """
    PixelRNN model using Row-LSTM residual blocks for autoregressive generation.

    Architecture:

        MaskedConv(A) -> n_block x ResidualRowLSTMBlock -> 1x1 Conv -> logits

    Args:
        n_channel (int): Number of image channels (1 for MNIST, 3 for CIFAR-10).
        h (int): Hidden dimension (the masked conv produces 2*h channels).
        n_block (int): Number of ResidualRowLSTMBlock layers.
        discrete_channel (int): Number of discrete intensity values (default 256).
    """

    def __init__(self, n_channel: int = 3, h: int = 32,
                 n_block: int = 12, discrete_channel: int = 256):
        super().__init__()
        self.discrete_channel = discrete_channel

        self.MaskAConv = MaskedConv('A', n_channel, 2 * h, kernel_size=7, stride=1, pad=3)

        block_residual = []
        for _ in range(n_block):
            block_residual.append(ResidualRowLSTMBlock(2 * h))
        self.block_residual = nn.Sequential(*block_residual)

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2 * h, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, n_channel * discrete_channel, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PixelRNN.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (B, C, H, W, 256).
        """
        batch_size, c_in, height, width = x.size()

        x = self.MaskAConv(x)          
        x = self.block_residual(x)      
        x = self.out(x)                 
        x = x.reshape(batch_size, c_in, self.discrete_channel, height, width)
        x = x.permute(0, 1, 3, 4, 2)   
        return x


class GatedPixelCNN(nn.Module):
    """
    Gated PixelCNN with vertical and horizontal stacks.

    Eliminates the blind spot of the original PixelCNN by separating
    vertical context (rows above) from horizontal context (pixels to
    the left on the current row) and combining them via a learned projection.
    A shared skip accumulator is passed through all blocks and used
    for the final output projection.

    Args:
        n_channel (int): Number of image channels.
        h (int): Number of feature channels throughout the network.
        n_block (int): Number of GatedPixelCNNBlock layers.
    """

    def __init__(self, n_channel: int = 3, h: int = 64, n_block: int = 12):
        super().__init__()
        self.n_channel = n_channel

        self.input_conv = MaskedConv('A', n_channel, h,
                                     kernel_size=7, stride=1, pad=3)

        self.blocks = nn.ModuleList(
            [GatedPixelCNNBlock(h) for _ in range(n_block)]
        )
        self.out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(h, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, n_channel * 256, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GatedPixelCNN.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (B, C, H, W, 256).
        """
        v = self.input_conv(x)         
        h = v.clone()
        skip = torch.zeros_like(h)

        for block in self.blocks:
            v, h, skip = block(v, h, skip)

        x = self.out(skip)             
        B, _, H, W = x.shape
        x = x.reshape(B, self.n_channel, 256, H, W)
        x = x.permute(0, 1, 3, 4, 2)  
        return x


class ConditionalPixelCNN(PixelCNN):
    """
    PixelCNN decoder conditioned on a latent vector z.

    Extends PixelCNN by projecting z onto the feature space and adding
    it after the first masked convolution block.

    Args:
        n_channel (int): Number of image channels.
        h (int): Bottleneck dimension.
        latent_dim (int): Dimensionality of the conditioning latent vector.
        discrete_channel (int): Number of discrete intensity values (default 256).
    """

    def __init__(self, n_channel: int = 3, h: int = 32,
                 latent_dim: int = 128, discrete_channel: int = 256):
        super().__init__(n_channel, h, discrete_channel)
        self.z_proj = nn.Linear(latent_dim, 2 * h)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass conditioned on latent vector z.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            z (torch.Tensor): Latent vector of shape (B, latent_dim).

        Returns:
            torch.Tensor: Output logits of shape (B, C, H, W, 256).
        """
        batch_size, c_in, height, width = x.size()
        cond = self.z_proj(z).unsqueeze(-1).unsqueeze(-1)  

        x = self.first_block(x)
        x = x + cond

        for block in self.res_blocks:
            x = block(x)

        x = self.final_block(x)
        x = x.reshape(batch_size, c_in, self.discrete_channel, height, width)
        x = x.permute(0, 1, 3, 4, 2)   
        return x


class PixelCNNAutoencoder(nn.Module):
    """
    Full autoencoder combining a convolutional Encoder
    with a ConditionalPixelCNN decoder.

    The encoder maps an input image to a latent vector z;
    the decoder reconstructs the image autoregressively conditioned on z.
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = ConditionalPixelCNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode x to z, then decode autoregressively conditioned on z.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Reconstructed image logits of shape (B, C, H, W, 256).
        """
        z = self.encoder(x)
        out = self.decoder(x, z)
        return out