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
    
class RowLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            4 * hidden_channels,
            kernel_size=(3,1),
            padding=(1,0)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = torch.zeros(B, hidden_channels, H, W)
        c = torch.zeros_like(h)

        for j in range(W):
            gates = self.conv(x[:,:,:,j:j+1])
            i,f,o,g = torch.chunk(gates,4,dim=1)

            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)

            c = f*c + i*g
            h = o*torch.tanh(c)

        return h
    
class ResidualRowLSTMBlock(nn.Module):
    def __init__(self, in_out_channels):
        super().__init__()

        # Row LSTM: input in_out_channels, output in_out_channels // 2
        # This allows the conv1x1 to expand it back to in_out_channels
        internal_channels = in_out_channels // 2
        self.row_lstm = RowLSTM(in_channels=in_out_channels, hidden_channels=internal_channels)

        # Conv 1x1: internal_channels -> in_out_channels
        self.conv1x1 = nn.Conv2d(internal_channels, in_out_channels, kernel_size=1)

    def forward(self, x):

        residual = x

        x = self.row_lstm(x)
        x = self.conv1x1(x)

        return x + residual
    
class VerticalStack(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            2*out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding)
        )

        self.gate = GatedActivation()

    def forward(self, x):

        x = self.conv(x)
        x = self.gate(x)

        return x
    
class HorizontalStack(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            2*out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding)
        )

        self.gate = GatedActivation()

        self.v_to_h = nn.Conv2d(out_channels, 2*out_channels, 1)

        self.residual = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, h, v):

        h_out = self.conv(h)

        v_proj = self.v_to_h(v)

        h_out = h_out + v_proj

        h_out = self.gate(h_out)

        return self.residual(h_out) + h
    
class GatedPixelCNNBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.vertical = VerticalStack(channels, channels)
        self.horizontal = HorizontalStack(channels, channels)

    def forward(self, v, h):

        v_out = self.vertical(v)

        h_out = self.horizontal(h, v_out)

        return v_out, h_out
    
class Encoder(nn.Module):

    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(64,128,4,2,1),
            nn.ReLU(),

            nn.Conv2d(128,256,4,2,1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(256, latent_dim)

    def forward(self,x):

        x = self.net(x)

        x = x.view(x.size(0),-1)

        z = self.fc(x)

        return z
    
class ConditionalPixelCNN(PixelCNN):  # decoder autoregresive

    def __init__(self, n_channel=3, h=32, latent_dim=128, discrete_channel=256):

        super().__init__(n_channel, h, discrete_channel)

        self.z_proj = nn.Linear(latent_dim, 2*h)

    def forward(self, x, z):

        batch_size, c_in, height, width = x.size()

        cond = self.z_proj(z).unsqueeze(-1).unsqueeze(-1)

        x = self.MaskAConv(x)

        x = x + cond

        x = self.MaskBConv(x)

        x = self.out(x)

        x = x.view(batch_size, c_in, self.discrete_channel, height, width)

        x = x.permute(0,1,3,4,2)

        return x