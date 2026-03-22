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

    git clone https://github.com/JuancarlosPG2004/Pixel-Neural-Networks.git
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
| `--n_epochs` | Number of epochs | `20` |
| `--batch_size` | Batch size | `32` |
| `--h` | Bottleneck dimension | `128` |
| `--n_block` | Number of residual blocks | `10` |
| `--lr` | Learning rate | `1e-3` |
| `--optimizer` | PyTorch optimizer (`Adam`, `AdamW`, `RMSprop`) | `Adam` |

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
- **MNIST**: Grayscale images 28×28, 10 classes

---

## Trained Models

### PixelCNN — MNIST

| Parameter | Value |
|-----------|-------|
| `n_epochs` | |
| `batch_size` | |
| `h` | |
| `n_block` | |
| `lr` | |
| `optimizer` | |

### PixelCNN — CIFAR-10

| Parameter | Value |
|-----------|-------|
| `n_epochs` | |
| `batch_size` | |
| `h` | |
| `n_block` | |
| `lr` | |
| `optimizer` | |

---

## Known Issues and Limitations

### PixelRNN — poor generation quality
Despite multiple training runs and hyperparameter tuning, we were unable to obtain a PixelRNN model that generates visually coherent images. Training is stable and loss decreases, but the sampled images remain noisy and lack meaningful structure. We attribute this to the computational constraints of the Row-LSTM approach, which requires sequential processing and is very sensitive to the number of layers and the hidden dimension `h`. Longer training with larger models would likely be needed to achieve competitive results.

### GatedPixelCNN — unstable training
The GatedPixelCNN architecture also proved difficult to train reliably. We observed that loss occasionally diverges mid-training, and the generated samples are inconsistent across runs. The vertical and horizontal stacks are sensitive to weight initialization and learning rate, and we did not have sufficient time to fully stabilize the training procedure.

### ConditionalPixelCNN and PixelCNNAutoencoder — not trained
The `ConditionalPixelCNN` and `PixelCNNAutoencoder` classes are fully implemented in `architecture.py` and `model.py`, but we did not have time to train or evaluate them. The autoencoder combines a convolutional encoder that maps an image to a latent vector `z` with a conditional PixelCNN decoder that reconstructs the image autoregressively conditioned on `z`. This is a promising direction for future work.

---

## Bibliography

- van den Oord, A., Kalchbrenner, N., and Kavukcuoglu, K. *Pixel Recurrent Neural Networks*. Proceedings of the 33rd International Conference on Machine Learning (ICML), New York, NY, USA, 2016. JMLR: W&CP volume 48. arXiv:1601.06759.

- van den Oord, A., Kalchbrenner, N., Vinyals, O., Espeholt, L., Graves, A., and Kavukcuoglu, K. *Conditional Image Generation with PixelCNN Decoders*. Advances in Neural Information Processing Systems (NeurIPS), 2016. arXiv:1606.05328.
