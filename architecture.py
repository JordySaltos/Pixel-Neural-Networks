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
        with torch.no_grad():
            self.weight.data.copy_(self.weight.data * self.mask)
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
            nn.ReLU(inplace=True),          
            nn.Conv2d(2 * h, h, kernel_size=1),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            MaskedConv('B', h, h, k_size, stride, pad),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, 2 * h, kernel_size=1),
            nn.BatchNorm2d(2 * h),
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
    """Row-LSTM unit that processes one column at a time left-to-right.

    For each column j the hidden state is updated using a (3,1) convolution
    over a causal vertical context (current row and the row above), keeping
    causality in both spatial dimensions.

    Args:
        in_channels: Number of input feature channels.
        hidden_channels: Number of LSTM hidden (output) channels.
    """

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels,
            4 * hidden_channels,
            kernel_size=(3, 1),
            padding=(0, 0),  
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Row-LSTM over all columns and return the full feature map."""
        B, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_channels, H, 1, device=x.device)
        c = torch.zeros_like(h)

        outputs = []
        for j in range(W):
            col = nn.functional.pad(x[:, :, :, j:j+1], (0, 0, 2, 0))
            gates = self.conv(col)
            i,f,o,g = torch.chunk(gates,4,dim=1)

            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)

            c = f*c + i*g
            h = o*torch.tanh(c)
            outputs.append(h)

        return torch.cat(outputs, dim=3)
    
class ResidualRowLSTMBlock(nn.Module):
    """Residual block wrapping a RowLSTM with a 1×1 projection.

    The bottleneck halves the channels internally, then a 1×1 conv restores
    the original width before adding the skip connection.

    Args:
        in_out_channels: Number of channels for both input and output.
    """

    def __init__(self, in_out_channels):
        super().__init__()

        internal_channels = in_out_channels // 2
        self.row_lstm = RowLSTM(in_channels=in_out_channels, hidden_channels=internal_channels)

        # Conv 1x1: internal_channels -> in_out_channels
        self.conv1x1 = nn.Conv2d(internal_channels, in_out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Row-LSTM bottleneck and add the residual skip connection."""
        residual = x

        x = self.row_lstm(x)
        x = self.conv1x1(x)

        return x + residual
    

class GatedActivation(nn.Module):
    """Element-wise gated activation: tanh(a) * sigmoid(b).

    Splits the channel dimension in half and applies tanh to the first half
    and sigmoid to the second, then multiplies them element-wise.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split channels and apply gated activation."""
        a, b = torch.chunk(x, 2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)
    
class VerticalStack(nn.Module):
    """Causal vertical convolution stack for GatedPixelCNN.

    Pads (kernel_size - 1) rows on the top only so the convolution cannot
    attend to any row below the current position.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (before gating; the conv
            produces ``2 * out_channels`` which GatedActivation halves).
        kernel_size: Spatial kernel size (default 3).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.pad_top = kernel_size - 1
        self.pad_h   = kernel_size // 2   # symmetric horizontal pad

        self.conv = nn.Conv2d(
            in_channels,
            2 * out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(0, self.pad_h),  # vertical handled manually
        )

        self.gate = GatedActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal top-padding, convolution, and gated activation."""
        # Pad only the top — convolution cannot see rows below the current one
        x = nn.functional.pad(x, (0, 0, self.pad_top, 0))
        x = self.conv(x)
        x = self.gate(x)
        return x

class HorizontalStack(nn.Module):
    """Causal horizontal convolution stack for GatedPixelCNN.

    Pads (kernel_size - 1) columns on the left only so the convolution
    cannot attend to any pixel to the right of the current position.
    Also projects the vertical-stack output and adds it before gating.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial kernel size (default 3).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.pad_left = kernel_size - 1

        self.conv = nn.Conv2d(
            in_channels,
            2 * out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, 0),  # horizontal handled manually
        )

        self.gate = GatedActivation()

        self.v_to_h  = nn.Conv2d(out_channels, 2 * out_channels, 1)
        self.residual = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply causal left-padding, fuse vertical output, gate, and add residual."""
        # Pad only the left — convolution cannot see pixels to the right
        h_out = nn.functional.pad(h, (self.pad_left, 0, 0, 0))
        h_out = self.conv(h_out)

        v_proj = self.v_to_h(v)

        h_out = h_out + v_proj
        h_out = self.gate(h_out)

        return self.residual(h_out) + h
    
class GatedPixelCNNBlock(nn.Module):
    """One layer of a Gated PixelCNN combining vertical and horizontal stacks.

    Args:
        channels: Number of feature channels (same for input and output).
    """

    def __init__(self, channels):
        super().__init__()

        self.vertical = VerticalStack(channels, channels)
        self.horizontal = HorizontalStack(channels, channels)

    def forward(
        self, v: torch.Tensor, h: torch.Tensor
    ) -> tuple:
        """Run one gated block; returns updated (v_out, h_out)."""
        v_out = self.vertical(v)

        h_out = self.horizontal(h, v_out)

        return v_out, h_out
    
class Encoder(nn.Module):
    """Convolutional encoder that maps an image to a latent vector.

    Used as the conditioning encoder in PixelCNNAutoencoder.

    Args:
        in_channels: Number of image channels.
        latent_dim: Dimensionality of the output latent vector.
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image batch to latent vectors of shape (B, latent_dim)."""
        x = self.net(x)

        x = x.view(x.size(0),-1)

        z = self.fc(x)

        return z