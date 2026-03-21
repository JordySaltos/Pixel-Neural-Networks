#  Pixel Neural Networks

Autoregressive image generation models implemented in PyTorch: **PixelCNN**, **PixelRNN**, and **GatedPixelCNN**, trained on **CIFAR-10** and **MNIST**.

---

##  Description

Autoregressive models generate images pixel by pixel, modeling the joint distribution of all pixels as a product of conditional distributions. Each pixel is predicted based on all previous pixels in raster scan order (left to right, top to bottom).

This project implements three variants:

- **PixelCNN** — uses masked convolutions with residual blocks to efficiently capture causal context.
- **PixelRNN** — uses Row-LSTM cells to model sequential dependencies between pixels.
- **GatedPixelCNN** — eliminates the blind spot of the original PixelCNN by separating context into vertical and horizontal stacks connected via gated activations.

---

##  Project Structure

    Pixel-Neural-Networks/
    │
    ├── architecture.py      # Building blocks: MaskedConv, ResidualBlock, RowLSTM, GatedPixelCNNBlock...
    ├── model.py             # Full models: PixelCNN, PixelRNN, GatedPixelCNN, ConditionalPixelCNN
    ├── train.py             # Solver: training loop, evaluation and sampling
    ├── Loader.py            # DataLoaders for CIFAR-10 and MNIST
    ├── Configuration.py     # Hyperparameter configuration via CLI
    ├── main.py              # Main entry point
    │
    ├── dataset/             # Automatically downloaded datasets
    ├── results/             # Checkpoints and samples generated per epoch
    └── docs/                # Sphinx-generated documentation

---

##  Installation

    git clone https://github.com/JordySaltos/Pixel-Neural-Networks.git
    cd Pixel-Neural-Networks
    pip install torch torchvision tqdm numpy

---

##  Usage

### Training

    python main.py --model_type PixelCNN --dataset CIFAR10 --n_epochs 20
    python main.py --model_type GatedPixelCNN --dataset MNIST --n_epochs 15 --h 64
    python main.py --model_type PixelRNN --dataset CIFAR10 --batch_size 4 --n_block 6

### Available Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_type` | Architecture: `PixelCNN`, `PixelRNN`, `GatedPixelCNN` | `PixelCNN` |
| `--dataset` | Dataset: `CIFAR10` or `MNIST` | `CIFAR10` |
| `--n_epochs` | Number of epochs | `5` |
| `--batch_size` | Batch size | `8` |
| `--h` | Bottleneck dimension | `128` |
| `--n_block` | Number of residual blocks | `15` |
| `--lr` | Learning rate | `1e-3` |
| `--optimizer` | PyTorch optimizer | `RMSprop` |

---

##  Architectures

### PixelCNN
    Input -> MaskedConv(A) -> [ResidualBlock(B) x N] -> FinalBlock -> Logits [B, C, H, W, 256]
    Each ResidualBlock follows a bottleneck pattern: 2h -> h (1×1) -> h (3×3 masked) -> 2h (1×1)

### PixelRNN
    Input -> MaskedConv(A) -> [ResidualRowLSTMBlock x N] -> Conv1×1 -> Logits [B, C, H, W, 256]
    Processes column by column while maintaining an LSTM hidden state to capture vertical dependencies

### GatedPixelCNN
    Input -> MaskedConv(A) -> [VerticalStack + HorizontalStack x N] -> Conv1×1 -> Logits [B, C, H, W, 256]
    Vertical stack captures context from rows above; horizontal stack captures pixels to the left.
    Both fused via tanh(a) * sigmoid(b) gated activations

---

##  Training Details

- **Loss**: Cross-Entropy over 256 intensity levels per channel
- **Scheduler**: ReduceLROnPlateau (factor 0.5, patience 3)
- **Sampling**: Images generated at the end of each epoch and saved to `results/`
- **Checkpoints**: Weights saved to `results/<model>_<dataset>_<timestamp>/model_weights.pth`

---

##  Documentation

Full API documentation generated with Sphinx from source docstrings.

    cd docs
    .\make.bat html
    .\make.bat latexpdf

---

##  Datasets

Datasets are downloaded automatically via torchvision when training starts:

- **CIFAR-10**: RGB images 32×32, 10 classes
- **MNIST**: Grayscale images 28×28 -> zero-padded to 32×32