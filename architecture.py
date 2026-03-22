import torch
import torch.nn as nn
import torch.nn.functional as F


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
        mask_type (str): 'A' or 'B'.
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        kernel_size (int): Kernel size.
        stride (int): Stride.
        pad (int): Padding.
    """

    def __init__(
        self, mask_type: str, c_in: int, c_out: int,
        kernel_size: int, stride: int, pad: int
    ):
        super().__init__(c_in, c_out, kernel_size, stride, pad, bias=False)
        assert mask_type in ('A', 'B'), \
            f"mask_type must be 'A' or 'B', got '{mask_type}'"
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
        """
        Apply masked convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W).
        """
        return F.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class FirstBlock(nn.Module):
    """
    First PixelCNN block using type-A masked convolution.

    Applies a 7x7 type-A MaskedConv followed by BatchNorm.
    Used as the entry point of the network to ensure no pixel
    attends to itself or future pixels.

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        kernel_size (int): Kernel size (default 7).
        stride (int): Stride (default 1).
        pad (int): Padding (default 3).
    """

    def __init__(
        self, c_in: int = 3, c_out: int = 256,
        kernel_size: int = 7, stride: int = 1, pad: int = 3
    ):
        super().__init__()
        self.block = nn.Sequential(
            MaskedConv('A', c_in, c_out, kernel_size, stride, pad),
            nn.BatchNorm2d(c_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the first block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W).
        """
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual PixelCNN block with bottleneck pattern.

    Architecture:

        2h -> h (1x1) -> h (3x3 masked B) -> 2h (1x1)

    Includes residual skip connection.

    Args:
        h (int): Bottleneck dimension.
        kernel_size (int): Kernel size for the masked convolution (default 3).
        stride (int): Stride (default 1).
        pad (int): Padding (default 1).
    """

    def __init__(self, h: int = 128, kernel_size: int = 3, stride: int = 1, pad: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
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
        """
        Forward pass through the residual block with skip connection.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 2h, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, 2h, H, W).
        """
        return x + self.block(x)


class FinalBlock(nn.Module):
    """
    Final PixelCNN block projecting features to output logits.

    Projects the feature map to n_channel * discrete_channel channels
    using two 1x1 convolutions with ReLU and BatchNorm in between.

    Args:
        n_channel (int): Number of image channels.
        h (int): Bottleneck dimension.
        discrete_channel (int): Number of discrete values per channel (default 256).
    """

    def __init__(self, n_channel: int = 3, h: int = 128, discrete_channel: int = 256):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * h, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, n_channel * discrete_channel, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the final projection block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 2h, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, n_channel * discrete_channel, H, W).
        """
        return self.block(x)


class RowLSTM(nn.Module):
    """
    Row-LSTM: processes one row at a time top-to-bottom,
    maintaining causal context (no future rows visible).

    Uses a (1, kernel_width) input conv + 1x1 hidden-to-hidden conv.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden/output channels.
        kernel_width (int): Width of the input-to-hidden convolution kernel (default 3).
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_width: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad_w = kernel_width // 2

        self.conv_i2h = nn.Conv2d(
            in_channels, 4 * hidden_channels,
            kernel_size=(1, kernel_width),
            padding=(0, pad_w),
        )
        self.conv_h2h = nn.Conv2d(
            hidden_channels, 4 * hidden_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Row-LSTM over all rows.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, hidden_channels, H, W).
        """
        B, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_channels, 1, W, device=x.device)
        c = torch.zeros_like(h)
        outputs = []

        for i in range(H):
            row = x[:, :, i:i + 1, :]                          # (B, C, 1, W)
            gates = self.conv_i2h(row) + self.conv_h2h(h)
            ig, fg, og, gg = torch.chunk(gates, 4, dim=1)

            ig = torch.sigmoid(ig)
            fg = torch.sigmoid(fg)
            og = torch.sigmoid(og)
            gg = torch.tanh(gg)

            c = fg * c + ig * gg
            h = og * torch.tanh(c)
            outputs.append(h)                                   # (B, hidden, 1, W)

        return torch.cat(outputs, dim=2)                        # (B, hidden, H, W)


class ResidualRowLSTMBlock(nn.Module):
    """
    Residual Row-LSTM block with 1x1 projection.

    Applies a RowLSTM with internal bottleneck (in_out_channels // 2)
    followed by a 1x1 convolution to restore the channel dimension,
    and adds a residual skip connection.

    Args:
        in_out_channels (int): Number of input and output channels.
    """

    def __init__(self, in_out_channels: int):
        super().__init__()
        internal_channels = in_out_channels // 2
        self.row_lstm = RowLSTM(
            in_channels=in_out_channels,
            hidden_channels=internal_channels,
        )
        self.conv1x1 = nn.Conv2d(internal_channels, in_out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        residual = x
        x = self.row_lstm(x)
        x = self.conv1x1(x)
        return x + residual


class GatedActivation(nn.Module):
    """
    Gated activation: tanh(a) * sigmoid(b).

    Splits channels in half along dim=1, applies tanh to the first half
    and sigmoid to the second half, then multiplies element-wise.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gated activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 2C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
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
        super().__init__()
        self.pad_top = kernel_size - 1
        self.pad_h   = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            2 * out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(0, self.pad_h),
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")
        nn.init.zeros_(self.conv.bias)
        self.conv.weight.data *= 0.5

        self.gate = GatedActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal top-padding, convolution, and gated activation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        H = x.shape[2]
        x = F.pad(x, (0, 0, self.pad_top, 0))
        x = self.conv(x)
        x = x[:, :, :H, :]    
        x = self.gate(x)
        return x


class HorizontalStack(nn.Module):
    """
    Causal horizontal convolution: pads (kernel_size-1) columns on the left.
    Fuses the vertical-stack output before gating.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Spatial kernel size (default 3).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.pad_left = kernel_size - 1

        self.conv = nn.Conv2d(
            in_channels,
            2 * out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, 0),
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")
        nn.init.zeros_(self.conv.bias)
        self.conv.weight.data *= 0.5

        self.gate = GatedActivation()
        self.v_to_h = nn.Conv2d(out_channels, 2 * out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.v_to_h.weight, nonlinearity="linear")
        nn.init.zeros_(self.v_to_h.bias)
        self.v_to_h.weight.data *= 0.5

        self.residual = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.residual.weight, nonlinearity="linear")
        nn.init.zeros_(self.residual.bias)

    def forward(self, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply causal left-padding, fuse vertical output, gate, and add residual.

        Args:
            h (torch.Tensor): Horizontal input feature map of shape (B, C, H, W).
            v (torch.Tensor): Vertical feature map from VerticalStack of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output feature map of shape (B, out_channels, H, W).
        """
        h_out = F.pad(h, (self.pad_left, 0, 0, 0))
        h_out = self.conv(h_out)
        v_proj = self.v_to_h(v)
        h_out = self.gate(h_out + v_proj)
        return self.residual(h_out) + h


class GatedPixelCNNBlock(nn.Module):
    """
    One layer of a Gated PixelCNN combining vertical and horizontal stacks.

    Processes the vertical and horizontal feature maps jointly,
    accumulates a skip connection output, and returns updated streams.

    Args:
        channels (int): Number of feature channels (same for input and output).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.vertical   = VerticalStack(channels, channels)
        self.horizontal = HorizontalStack(channels, channels)
        self.h_skip = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self, v: torch.Tensor, h: torch.Tensor, skip: torch.Tensor
    ) -> tuple:
        """
        Run one gated block and update the skip accumulator.

        Args:
            v (torch.Tensor): Vertical input of shape (B, C, H, W).
            h (torch.Tensor): Horizontal input of shape (B, C, H, W).
            skip (torch.Tensor): Accumulated skip tensor of shape (B, C, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Updated (v_out, h_out, skip), each of shape (B, C, H, W).
        """
        v_out = self.vertical(v)
        v_out = v_out + v          
        h_out = self.horizontal(h, v_out)
        skip  = skip + self.h_skip(h_out)
        return v_out, h_out, skip


class Encoder(nn.Module):
    """
    Convolutional encoder mapping an image to a latent vector.

    Used as the conditioning encoder in PixelCNNAutoencoder.

    Args:
        in_channels (int): Number of input image channels.
        latent_dim (int): Dimensionality of the output latent vector.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),          nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),         nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image batch to latent vectors.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Latent vectors of shape (B, latent_dim).
        """
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)