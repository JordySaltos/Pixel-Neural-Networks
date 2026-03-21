import torch
import torch.nn as nn

class MaskedConv(nn.Conv2d):
    """
    2D convolution with causal masking for PixelCNN.

    Implements two mask types:

    - Type 'A': excludes the center pixel and all subsequent pixels
      in raster scan order.
      Used in the first layer to ensure predictions do not depend on themselves.
    - Type 'B': includes the center pixel but excludes subsequent ones.
      Used in residual layers to allow self-connection.

    Args:
        mask_type (str): 'A' or 'B'
        c_in (int): Number of input channels
        c_out (int): Number of output channels
        kernel_size (int): Kernel size
        stride (int): Stride
        pad (int): Padding
    """

    def __init__(
        self, mask_type: str, c_in: int, c_out: int,
        kernel_size: int, stride: int, pad: int
    ):
        """Initialise the masked convolution and register the causal mask buffer."""
        super().__init__(c_in, c_out, kernel_size, stride, pad, bias=False)
        assert mask_type in ('A', 'B'), (
            f"mask_type must be 'A' or 'B', got '{mask_type}'"
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
        """Apply masked convolution to input tensor x."""
        self.weight.data.mul_(self.mask)
        return super().forward(x)


class FirstBlock(nn.Module):
    """
    First PixelCNN block using type-A masked convolution.

    Args:
        c_in (int): Input channels
        c_out (int): Output channels
        kernel_size (int): Kernel size
        stride (int): Stride
        pad (int): Padding
    """

    def __init__(
        self, c_in: int = 3, c_out: int = 256,
        kernel_size: int = 7, stride: int = 1, pad: int = 3
    ):
        """Initialise the type-A masked convolution and batch normalisation."""
        super().__init__()
        self.block = nn.Sequential(
            MaskedConv('A', c_in, c_out, kernel_size, stride, pad),
            nn.BatchNorm2d(c_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through first block."""
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual PixelCNN block with bottleneck pattern.

    Architecture:

        2h -> h (1x1) -> h (3x3 masked B) -> 2h (1x1)

    Includes residual skip connection.

    Args:
        h (int): Bottleneck dimension
        kernel_size (int): Kernel size
        stride (int): Stride
        pad (int): Padding
    """

    def __init__(self, h: int = 128, kernel_size: int = 3, stride: int = 1, pad: int = 1):
        """Initialise the bottleneck residual block layers."""
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * h, h, kernel_size=1),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            MaskedConv('B', h, h, kernel_size, stride, pad),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, 2 * h, kernel_size=1),
            nn.BatchNorm2d(2 * h)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block with skip connection."""
        return x + self.block(x)


class FinalBlock(nn.Module):
    """
    Final PixelCNN block projecting features to output logits.

    Args:
        n_channel (int): Input channels
        h (int): Bottleneck dimension
        discrete_channel (int): Number of discrete values per channel
    """

    def __init__(self, n_channel: int = 3, h: int = 128, discrete_channel: int = 256):
        """Initialise the output projection layers."""
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * h, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, n_channel * discrete_channel, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through final block."""
        return self.block(x)


class RowLSTM(nn.Module):
    """
    Row-LSTM for PixelRNN.

    Processes one column at a time, maintaining causal context vertically.

    Args:
        in_channels (int): Input channels
        hidden_channels (int): Hidden/output channels
    """

    def __init__(self, in_channels, hidden_channels):
        """Initialise the causal (3,1) convolution used for gate computation."""
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels, 4 * hidden_channels,
            kernel_size=(3, 1), padding=(0, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Row-LSTM over all columns.

        Args:
            x (Tensor): Input of shape [B, C, H, W]

        Returns:
            Tensor: Output of shape [B, hidden_channels, H, W]
        """
        B, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_channels, H, 1, device=x.device)
        c = torch.zeros_like(h)
        outputs = []

        for j in range(W):
            col = nn.functional.pad(x[:, :, :, j:j+1], (0,0,2,0))
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
    """
    Residual Row-LSTM block with 1x1 projection.

    Args:
        in_out_channels (int): Input/output channels
    """

    def __init__(self, in_out_channels):
        """Initialise the RowLSTM and the 1x1 projection convolution."""
        super().__init__()
        internal_channels = in_out_channels // 2
        self.row_lstm = RowLSTM(
            in_channels=in_out_channels, hidden_channels=internal_channels
        )
        self.conv1x1 = nn.Conv2d(internal_channels, in_out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x (Tensor): Input tensor [B, C, H, W]

        Returns:
            Tensor: Output tensor [B, C, H, W]
        """
        residual = x
        x = self.row_lstm(x)
        x = self.conv1x1(x)
        return x + residual


class GatedActivation(nn.Module):
    """
    Gated activation: tanh(a) * sigmoid(b)

    Splits channels in half, applies activations, multiplies element-wise.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated activation to input tensor."""
        a, b = torch.chunk(x, 2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)



class VerticalStack(nn.Module):
    """
    Causal vertical convolution stack for GatedPixelCNN.

    Pads (kernel_size - 1) rows on the top only so the convolution cannot
    attend to any row below the current position.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (before gating; the conv
            produces 2 * out_channels which GatedActivation halves).
        kernel_size (int): Spatial kernel size (default 3).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """Initialise padding constants and the gated convolution."""
        super().__init__()
        self.pad_top = kernel_size - 1
        self.pad_h = kernel_size // 2  # symmetric horizontal padding

        self.conv = nn.Conv2d(
            in_channels,
            2 * out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(0, self.pad_h),  # vertical handled manually
        )

        self.gate = GatedActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal top-padding, convolution, and gated activation."""
        x = nn.functional.pad(x, (0, 0, self.pad_top, 0))
        x = self.conv(x)
        x = self.gate(x)
        return x


class HorizontalStack(nn.Module):
    """
    Causal horizontal convolution stack for GatedPixelCNN.

    Pads (kernel_size - 1) columns on the left only so the convolution
    cannot attend to pixels to the right of the current position.
    Also projects the vertical-stack output and adds it before gating.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Spatial kernel size (default 3).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """Initialise the causal horizontal convolution and projection layers."""
        super().__init__()
        self.pad_left = kernel_size - 1

        self.conv = nn.Conv2d(
            in_channels,
            2 * out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, 0)  # horizontal handled manually
        )

        self.gate = GatedActivation()
        self.v_to_h = nn.Conv2d(out_channels, 2 * out_channels, kernel_size=1)
        self.residual = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply causal left-padding, fuse vertical output, gate, and add residual.

        Args:
            h (Tensor): Horizontal input feature map [B, C, H, W]
            v (Tensor): Vertical feature map from VerticalStack [B, C, H, W]

        Returns:
            Tensor: Output feature map [B, C, H, W]
        """
        h_out = nn.functional.pad(h, (self.pad_left, 0, 0, 0))
        h_out = self.conv(h_out)
        v_proj = self.v_to_h(v)
        h_out = self.gate(h_out + v_proj)
        return self.residual(h_out) + h


class GatedPixelCNNBlock(nn.Module):
    """
    One layer of a Gated PixelCNN combining vertical and horizontal stacks.

    Args:
        channels (int): Number of feature channels (same for input and output).
    """

    def __init__(self, channels: int):
        """Initialise the vertical and horizontal stack sub-modules."""
        super().__init__()
        self.vertical = VerticalStack(channels, channels)
        self.horizontal = HorizontalStack(channels, channels)

    def forward(
        self, v: torch.Tensor, h: torch.Tensor
    ) -> tuple:
        """
        Run one gated block.

        Args:
            v (Tensor): Vertical input [B, C, H, W]
            h (Tensor): Horizontal input [B, C, H, W]

        Returns:
            tuple: Updated (v_out, h_out)
        """
        v_out = self.vertical(v)
        h_out = self.horizontal(h, v_out)
        return v_out, h_out


class Encoder(nn.Module):
    """
    Convolutional encoder mapping an image to a latent vector.

    Used as the conditioning encoder in PixelCNNAutoencoder.

    Args:
        in_channels (int): Number of input image channels.
        latent_dim (int): Dimensionality of the output latent vector.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        """Initialise the convolutional backbone and the linear projection."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image batch to latent vectors.

        Args:
            x (Tensor): Input images [B, C, H, W]

        Returns:
            Tensor: Latent vectors [B, latent_dim]
        """
        x = self.net(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z