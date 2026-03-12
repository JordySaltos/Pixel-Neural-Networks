import torch
import torch.nn as nn

class MaskedConv(nn.Conv2d):
    """
    2D Convolution with causal masking for PixelCNN.

    Implements two mask types:
    - Type 'A': Excludes the center pixel and all subsequent pixels in raster
      scan order. Used only in the first layer to ensure each pixel prediction
      does not depend on itself.
    - Type 'B': Includes the center pixel but excludes subsequent ones. Used
      in intermediate (residual) layers to allow self-connection.

    Args:
        mask_type (str): Mask type, either 'A' or 'B'.
        c_in (int): Input channels.
        c_out (int): Output channels.
        k_size (int): Kernel size.
        stride (int): Stride.
        pad (int): Padding.
    """

    def __init__(self, mask_type: str, c_in: int, c_out: int,
                 k_size: int, stride: int, pad: int):
        super().__init__(c_in, c_out, k_size, stride, pad, bias=False)

        assert mask_type in ('A', 'B'), (
            f"mask_type must be 'A' or 'B', got: '{mask_type}'"
        )
        self.mask_type = mask_type

        ch_out, ch_in, height, width = self.weight.size()
        mask = torch.ones(ch_out, ch_in, height, width)
        mask[:, :, height // 2 + 1:, :] = 0

        if mask_type == 'A':
            mask[:, :, height // 2, width // 2:] = 0
        else:
            mask[:, :, height // 2, width // 2 + 1:] = 0

        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class FirstBlock(nn.Module):
    """
    First block of the PixelCNN.

    Applies a type-A masked convolution

    Args:
        c_in (int): Input channels.
        c_out (int): Output channels.
        k_size (int): Kernel size.
        stride (int): Stride.
        pad (int): Padding.
    """

    def __init__(self, c_in: int = 3, c_out: int = 256,
                 k_size: int = 7, stride: int = 1, pad: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            MaskedConv('A', c_in, c_out, k_size, stride, pad),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual blocks of the PixelCNN.

    Implements a bottleneck pattern:
        2h → h (1×1) → h (3×3 masked B) → 2h (1×1)
    with a residual skip connection that adds the input to the output.

    Args:
        h (int): Bottleneck dimension.
        k_size (int): Kernel size.
        stride (int): Stride.
        pad (int): Padding.
    """

    def __init__(self, h: int = 128, k_size: int = 3,
                 stride: int = 1, pad: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(2 * h, h, kernel_size=1),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            MaskedConv('B', h, h, k_size, stride, pad),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, 2 * h, kernel_size=1),
            nn.BatchNorm2d(2 * h),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FinalBlock(nn.Module):
    """
    Final blocks of the PixelCNN.

    Args:
        n_channel (int): Input channels.
        h (int): Bottleneck dimension.
        discrete_channel (int): Number of discrete values per channel (default 256).
    """

    def __init__(self, n_channel: int = 3, h: int = 128,
                 discrete_channel: int = 256) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * h, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, n_channel * discrete_channel, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)