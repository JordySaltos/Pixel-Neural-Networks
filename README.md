# Pixel Neural Networks

Autoregressive image generation models implemented in PyTorch: **PixelCNN**, **PixelRNN**, and **GatedPixelCNN**, trained on **CIFAR-10** and **MNIST**.

---

## Description

Autoregressive models generate images pixel by pixel, modeling the joint distribution of all pixels as a product of conditional distributions. Each pixel is predicted based on all previous pixels in raster scan order (left to right, top to bottom).

This project implements three variants:

- **PixelCNN** — uses masked convolutions with residual blocks to efficiently capture causal context.
- **PixelRNN** — uses Row-LSTM cells to model sequential dependencies between pixels.
- **GatedPixelCNN** — eliminates the blind spot of the original PixelCNN by separating context into vertical and horizontal stacks connected via gated activations.

---

## Project Structure

```
Pixel-Neural-Networks/
│
├── architecture.py        # Building blocks: MaskedConv, ResidualBlock, RowLSTM, GatedPixelCNNBlock...
├── model.py               # Full models: PixelCNN, PixelRNN, GatedPixelCNN, ConditionalPixelCNN
├── train.py               # Solver and GatedSolver: training loop, evaluation and sampling
├── Loader.py              # DataLoaders for CIFAR-10 and MNIST
├── Configuration.py       # Hyperparameter configuration via CLI
├── main.py                # CLI entry point
├── app.py                 # Streamlit interactive application
├── download_weights.py    # Download pre-trained weights from Google Drive
│
├── dataset/               # Automatically downloaded datasets (not versioned)
└── results/               # Checkpoints and samples generated per run (not versioned)
```

---

## Installation

```bash
git clone https://github.com/JuancarlosPG2004/Pixel-Neural-Networks.git
cd Pixel-Neural-Networks
```

Install dependencies according to your setup:

```bash
# GPU (CUDA 12)
pip install -r requirements_gpu.txt

# CPU only
pip install -r requirements_cpu.txt
```

---

## Usage

### Streamlit app

```bash
streamlit run app.py
```

The app allows you to train models, generate images, and complete the bottom half of test images — all from the browser.

### CLI training

```bash
python main.py --model_type PixelCNN --dataset CIFAR10 --n_epochs 35
python main.py --model_type GatedPixelCNN --dataset CIFAR10 --n_epochs 15 --h 64
python main.py --model_type PixelRNN --dataset MNIST --n_epochs 20
```

### Available parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_type` | Architecture: `PixelCNN`, `PixelRNN`, `GatedPixelCNN` | `PixelCNN` |
| `--dataset` | Dataset: `CIFAR10` or `MNIST` | `CIFAR10` |
| `--n_epochs` | Number of epochs | `40` |
| `--batch_size` | Batch size | `64` |
| `--h` | Bottleneck dimension | `128` |
| `--n_block` | Number of residual blocks | `10` |
| `--lr` | Learning rate | `1e-3` |
| `--optimizer` | PyTorch optimizer (`Adam`, `AdamW`, `RMSprop`) | `Adam` |

---

## Architectures

### PixelCNN
```
Input -> MaskedConv(A) -> [ResidualBlock(B) x N] -> FinalBlock -> Logits [B, C, H, W, 256]
Each ResidualBlock: 2h -> h (1×1) -> h (3×3 masked) -> 2h (1×1)
```

### PixelRNN
```
Input -> MaskedConv(A) -> [ResidualRowLSTMBlock x N] -> Conv1×1 -> Logits [B, C, H, W, 256]
Processes row by row while maintaining an LSTM hidden state to capture vertical dependencies.
```

### GatedPixelCNN
```
Input -> MaskedConv(A) -> [VerticalStack + HorizontalStack x N] -> Conv1×1 -> Logits [B, C, H, W, 256]
Vertical stack captures context from rows above; horizontal stack captures pixels to the left.
Both fused via tanh(a) * sigmoid(b) gated activations.
```

---

## Documentation

Full API documentation generated with Sphinx from source docstrings.

```bash
cd docs
make html        # opens in docs/build/html/index.html
make latexpdf    # generates a PDF
make clean       # clear cached build before regenerating
```

---

## Training Details

- **Loss**: Cross-Entropy over 256 intensity levels per channel, reported in nats and bits/dim
- **Scheduler**: ReduceLROnPlateau (factor 0.5) — patience and gradient clipping tuned per dataset
- **Sampling**: Checkpoint images generated every 3 epochs and saved to `results/`
- **Checkpoints**: Weights saved to `results/<model>_<dataset>_<timestamp>/model_weights.pth`

---

## Pre-trained Weights

Weights are stored on Google Drive and can be downloaded with:

```bash
python download_weights.py
```

This restores the full `results/` folder locally. The Streamlit app detects available runs automatically — no additional configuration needed.

---

## Trained Models

### PixelCNN — MNIST

| Parameter | Value |
|-----------|-------|
| `n_epochs` | 40 |
| `batch_size` | 64 |
| `h` | 64 |
| `n_block` | 8 |
| `lr` | 3e-4 |
| `optimizer` | Adam |

### PixelCNN — CIFAR-10

| Parameter | Value |
|-----------|-------|
| `n_epochs` | 35 |
| `batch_size` | 32 |
| `h` | 64 |
| `n_block` | 7 |
| `lr` | 3e-4 |
| `optimizer` | Adam |

---

## Known Issues and Limitations

### PixelRNN — not evaluated

The PixelRNN architecture is fully implemented using Row-LSTM residual blocks, but we were not able to obtain a meaningful evaluation. The Row-LSTM requires sequential processing row by row, which makes training significantly slower than the convolutional models. Given our computational constraints, training for a sufficient number of epochs with an appropriate model size was not feasible within the project timeline. Its performance on MNIST and CIFAR-10 therefore remains unknown.

### GatedPixelCNN — black image collapse

During training, GatedPixelCNN consistently converged to a degenerate solution where the model assigned near-certain probability to black pixels (value 0) for every position. This is a known failure mode in autoregressive models trained on datasets where dark pixels are statistically dominant, such as CIFAR-10.

To address the all-black collapse, GatedPixelCNN uses a dedicated `GatedSolver` with targeted modifications: a linear LR warmup over the first 300 batches, `weight_decay=1e-5`, a guarded `ReduceLROnPlateau` that only activates once the loss drops below 4.0 nats, and a sampling temperature of 1.5. The all-black collapse was resolved, but the model currently converges to a uniform distribution over all 256 pixel values (loss ≈ ln(256) ≈ 5.55 nats), producing random coloured noise instead of structured images. This indicates the model is not learning any pixel dependencies — the gradients are flowing but carrying no signal. 

### ConditionalPixelCNN and PixelCNNAutoencoder — not trained

The `ConditionalPixelCNN` and `PixelCNNAutoencoder` classes are fully implemented in `model.py`, but were not trained or evaluated due to time constraints. The autoencoder combines a convolutional encoder that maps an image to a latent vector `z` with a conditional PixelCNN decoder that reconstructs the image autoregressively conditioned on `z`. This remains a direction for future work.

---

## Bibliography

- van den Oord, A., Kalchbrenner, N., and Kavukcuoglu, K. *Pixel Recurrent Neural Networks*. ICML 2016. arXiv:1601.06759.

- van den Oord, A., Kalchbrenner, N., Vinyals, O., Espeholt, L., Graves, A., and Kavukcuoglu, K. *Conditional Image Generation with PixelCNN Decoders*. NeurIPS 2016. arXiv:1606.05328.

### Implementation references

- PixelCNN implementation: https://github.com/j-min/PixelCNN
- GatedPixelCNN implementation: https://github.com/anordertoreclaim/PixelCNN
