"""The main module of the app.

Contains most of the functions governing the
different app modes.

"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from Configuration import BaseConfig
from Loader import get_loader, get_dataset_config, DATASET_CONFIGS
from model import PixelCNN, PixelRNN, GatedPixelCNN
from train import Solver


RESULTS_DIR = Path("results")
WEIGHTS_FILENAME = "model_weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Available models and their UI descriptions
MODEL_CONFIGS = {
    "PixelCNN": {
        "class": PixelCNN,
        "description": "Classic PixelCNN with masked convolutions and residual blocks. "
                       "Fast to train, good baseline.",
        "supports_h": True,
        "supports_n_block": True,
    },
    "PixelRNN": {
        "class": PixelRNN,
        "description": "PixelRNN with Row-LSTM units. Captures long-range dependencies "
                       "better than CNN but is significantly slower.",
        "supports_h": True,
        "supports_n_block": True,
    },
    "GatedPixelCNN": {
        "class": GatedPixelCNN,
        "description": "Gated PixelCNN with vertical + horizontal stacks. "
                       "Eliminates the blind spot of the original PixelCNN.",
        "supports_h": True,
        "supports_n_block": True,
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def find_latest_weights():
    """Return path to model_weights.pth in the most recent results folder."""
    if not RESULTS_DIR.exists():
        return None
    candidates = sorted(
        d / WEIGHTS_FILENAME
        for d in RESULTS_DIR.iterdir()
        if d.is_dir() and (d / WEIGHTS_FILENAME).exists()
    )
    return candidates[-1] if candidates else None


def find_sample_images():
    """Return epoch-*.png files from the most recent results folder."""
    if not RESULTS_DIR.exists():
        return []
    folders = sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
    if not folders:
        return []
    return sorted(folders[-1].glob("epoch-*.png"))


def read_config_from_results():
    """Read dataset, h and n_block from the config.txt of the latest run."""
    if not RESULTS_DIR.exists():
        return {}
    folders = sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
    if not folders:
        return {}
    config_file = folders[-1] / "config.txt"
    if not config_file.exists():
        return {}
    params = {}
    for line in config_file.read_text().splitlines():
        for key in ("dataset", "h", "n_block", "model_type"):
            if line.startswith(f"{key}:"):
                params[key] = line.split(":", 1)[1].strip()
    return params


@st.cache_resource
def load_model(weights_path: str, dataset: str, h: int, n_block: int,
               model_type: str = "PixelCNN"):
    n_channel = get_dataset_config(dataset)["n_channel"]
    model_class = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["PixelCNN"])["class"]

    if model_type == "PixelRNN":
        model = model_class(n_channel=n_channel, h=h).to(DEVICE)
    elif model_type == "GatedPixelCNN":
        model = model_class(in_channels=n_channel, channels=h, n_layers=n_block).to(DEVICE)
    else:  # PixelCNN (default)
        model = model_class(n_channel=n_channel, h=h, n_block=n_block).to(DEVICE)

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


# ── training ──────────────────────────────────────────────────────────────────

def run_training(dataset, n_epochs, batch_size, h, n_block, model_type):
    config = BaseConfig().initialize(
        parse=False,
        mode="train",
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer="RMSprop",
        dataset=dataset,
        h=h,
        n_block=n_block,
        model_type=model_type,
        log_interval=100,
        save_interval=10,
    )

    st.write(
        f"**Device:** `{DEVICE}` | **Dataset:** `{dataset}` | "
        f"**Model:** `{model_type}` | **h:** `{h}` | **n_block:** `{n_block}`"
    )
    st.write(f"**Results folder:** `{config.ckpt_dir}`")

    train_loader = get_loader(
        "./dataset", batch_size, train=True, dataset_name=dataset
    )
    test_loader = get_loader(
        "./dataset", batch_size, train=False, dataset_name=dataset
    )

    # Instantiate the chosen model
    n_channel = get_dataset_config(dataset)["n_channel"]
    model_class = MODEL_CONFIGS[model_type]["class"]
    if model_type == "PixelRNN":
        solver_model = model_class(n_channel=n_channel, h=h).to(DEVICE)
    elif model_type == "GatedPixelCNN":
        solver_model = model_class(in_channels=n_channel, channels=h, n_layers=n_block).to(DEVICE)
    else:
        solver_model = model_class(n_channel=n_channel, h=h, n_block=n_block).to(DEVICE)

    solver = Solver(config, train_loader, test_loader)
    solver.build(model_override=solver_model)

    progress = st.progress(0, text="Starting training…")
    loss_placeholder = st.empty()

    for epoch in range(1, n_epochs + 1):

        if epoch == 1:
            solver.sample(epoch)

        # train
        solver.model.train()
        batch_losses = []
        for images, _ in train_loader:
            images = images.to(DEVICE)
            logits = solver.model(images).contiguous().view(-1, 256)
            targets = (images.view(-1) * 255).long()
            loss = solver.criterion(logits, targets)
            solver.optimizer.zero_grad()
            loss.backward()
            solver.optimizer.step()
            batch_losses.append(float(loss.detach()))

        epoch_loss = float(np.mean(batch_losses))
        solver.train_losses.append(epoch_loss)

        # test
        test_loss = solver.test(epoch)
        solver.test_losses.append(test_loss)

        # sample
        solver.sample(epoch)

        # update UI
        progress.progress(epoch / n_epochs, text=f"Epoch {epoch}/{n_epochs}")
        loss_placeholder.write(
            f"**Epoch {epoch}** — train loss: `{epoch_loss:.4f}` | "
            f"test loss: `{test_loss:.4f}`"
        )

    # save weights
    weights_path = str(config.ckpt_dir / WEIGHTS_FILENAME)
    torch.save(solver.model.state_dict(), weights_path)
    st.success(f"Training complete! Weights saved to `{weights_path}`")

    _plot_losses(solver.train_losses, solver.test_losses)


def _plot_losses(train_losses, test_losses):
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(epochs, train_losses, marker="o", label="Train loss")
    ax.plot(epochs[:len(test_losses)], test_losses, marker="s", label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


# ── results viewer ────────────────────────────────────────────────────────────

def show_results():
    weights_path = find_latest_weights()

    if weights_path is None:
        st.warning("No trained model found. Go to **Train model** first.")
        return

    # Read architecture params saved during training
    saved = read_config_from_results()
    dataset    = saved.get("dataset")
    h          = int(saved.get("h", 128))
    n_block    = int(saved.get("n_block", 15))
    model_type = saved.get("model_type")

    st.write(
        f"Loaded weights from `{weights_path}`  \n"
        f"Dataset: `{dataset}` | Model: `{model_type}` | h: `{h}` | n_block: `{n_block}`"
    )

    model = load_model(str(weights_path), dataset, h, n_block, model_type)

    # Images saved during training
    sample_files = find_sample_images()
    if sample_files:
        st.subheader("Images generated during training")
        cols = st.columns(min(len(sample_files), 5))
        for col, img_path in zip(cols, sample_files[-5:]):
            col.image(str(img_path), caption=img_path.stem, use_container_width=True)

    # Generate new images on demand
    st.subheader("Generate new images")
    n_images = st.slider("Number of images to generate", 1, 16, 4)

    if st.button("Generate"):
        with st.spinner("Sampling pixel-by-pixel… (this can take a while)"):
            generated = _sample(model, dataset, n_images)

        grid = make_grid(generated.cpu(), nrow=4, normalize=True, pad_value=1)
        npimg = grid.numpy().transpose(1, 2, 0)
        # Grayscale images come out with 1 channel — squeeze for imshow
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]

        fig, ax = plt.subplots(figsize=(8, max(2, n_images // 4 * 2)))
        cmap = "gray" if get_dataset_config(dataset)["n_channel"] == 1 else None
        ax.imshow(np.clip(npimg, 0, 1), cmap=cmap)
        ax.axis("off")
        ax.set_title(f"PixelCNN generated samples ({dataset})")
        st.pyplot(fig)


def _sample(model, dataset, n_images):
    model.eval()
    ds_cfg = get_dataset_config(dataset)
    n_channel = ds_cfg["n_channel"]
    img_size  = ds_cfg["img_size"]

    generated = torch.zeros(n_images, n_channel, img_size, img_size, device=DEVICE)

    with torch.no_grad():
        for i in range(img_size):
            for j in range(img_size):
                output = model(generated)
                probs  = F.softmax(output[:, :, i, j], dim=2)
                for ch in range(n_channel):
                    pixel = (
                        torch.multinomial(probs[:, ch], 1).float() / 255.0
                    ).squeeze(-1)
                    generated[:, ch, i, j] = pixel

    return generated
# ── image completion experiment ───────────────────────────────────────────────

def show_completion():
    weights_path = find_latest_weights()

    if weights_path is None:
        st.warning("No trained model found. Go to **Train model** first.")
        return

    saved = read_config_from_results()
    dataset    = saved.get("dataset")
    h          = int(saved.get("h", 128))
    n_block    = int(saved.get("n_block", 15))
    model_type = saved.get("model_type")

    st.write(
        f"Loaded weights from `{weights_path}`  \n"
        f"Dataset: `{dataset}` | Model: `{model_type}` | h: `{h}` | n_block: `{n_block}`"
    )

    model = load_model(str(weights_path), dataset, h, n_block, model_type)

    st.subheader("Conditional Image Completion")
    st.write("Select a real image from the test dataset. We will mask the bottom half "
             "and let the model generate possible completions.")

# Load full test dataset
    full_loader = get_loader("./dataset", batch_size=10000, train=False, dataset_name=dataset)
    test_images, test_labels = next(iter(full_loader))

    col1, col2 = st.columns([1, 2])
    with col1:
        chosen_digit = st.selectbox("Select a digit (0–9)", list(range(10)))
        matching_indices = (test_labels == chosen_digit).nonzero(as_tuple=True)[0]

        if len(matching_indices) == 0:
            st.warning(f"No images found for digit {chosen_digit}.")
            return

        # Pick a random image among all examples of the chosen digit
        img_idx = matching_indices[torch.randint(len(matching_indices), (1,)).item()].item()
    with col2:
        n_completions = st.slider("Number of variations to generate", 1, 10, 1)

    real_image = test_images[img_idx:img_idx+1].to(DEVICE)

    if st.button("Autocomplete bottom half"):
        with st.spinner("Generating pixel by pixel... (this may take a while)"):
            completions = _sample_conditional(model, dataset, real_image, n_completions)

        ds_cfg = get_dataset_config(dataset)
        img_size = ds_cfg["img_size"]
        half_height = img_size // 2

        # Create the masked image for display
        masked_image = real_image.clone()
        masked_image[:, :, half_height:, :] = 0.0

        # Concatenate: [Masked, Gen 1, Gen 2, ..., Original]
        all_images = torch.cat([masked_image.cpu(), completions.cpu(), real_image.cpu()], dim=0)

        grid = make_grid(all_images, nrow=2 + n_completions, normalize=True, pad_value=1)
        npimg = grid.numpy().transpose(1, 2, 0)
        
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]

        fig, ax = plt.subplots(figsize=(12, 4))
        cmap = "gray" if ds_cfg["n_channel"] == 1 else None
        ax.imshow(np.clip(npimg, 0, 1), cmap=cmap)
        ax.axis("off")
        ax.set_title(f"Masked | {n_completions} Generated Completions | Original")
        st.pyplot(fig)


def _sample_conditional(model, dataset, real_image, n_completions):
    model.eval()
    ds_cfg = get_dataset_config(dataset)
    n_channel = ds_cfg["n_channel"]
    img_size  = ds_cfg["img_size"]
    half_height = img_size // 2

    # Duplicate the condition `n_completions` times
    generated = real_image.repeat(n_completions, 1, 1, 1).to(DEVICE)

    # Erase the bottom half so the model fills it
    generated[:, :, half_height:, :] = 0.0

    with torch.no_grad():
        # Start sampling from the half point
        for i in range(half_height, img_size):
            for j in range(img_size):
                output = model(generated)
                probs  = F.softmax(output[:, :, i, j], dim=2)
                for ch in range(n_channel):
                    pixel = (
                        torch.multinomial(probs[:, ch], 1).float() / 255.0
                    ).squeeze(-1)
                    generated[:, ch, i, j] = pixel

    return generated

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("PixelCNN — image generation")

    mode = st.sidebar.selectbox("Choose mode", ["Train model", "View results", "Image Completion"])

    if mode == "Train model":
        st.header("Train a Pixel Neural Network")

        dataset    = st.selectbox("Dataset", list(DATASET_CONFIGS.keys()))
        n_epochs   = st.slider("Number of epochs", 1, 150, 5)
        batch_size = st.selectbox("Batch size", [8, 16, 32, 64, 128], index=1)

        st.subheader("Model architecture")

        model_type = st.selectbox(
            "Model",
            list(MODEL_CONFIGS.keys()),
            help="Choose the autoregressive architecture to train.",
        )
        st.caption(MODEL_CONFIGS[model_type]["description"])

        h = st.slider(
            "h : bottleneck / channel dimension", 16, 256, 128, step=16,
            help=(
                "For PixelCNN / PixelRNN: bottleneck width of each residual block. "
                "For GatedPixelCNN: number of channels throughout the network. "
                "Larger = more capacity but slower."
            ),
        )

        n_block = st.slider(
            "n_block / n_layers : depth of the network", 4, 20, 10, 1,
            help="Number of residual / gated blocks. More = larger receptive field.",
        )

        ds_cfg = get_dataset_config(dataset)
        st.caption(
            f"Model channels: {ds_cfg['n_channel']} | "
            f"Image size: {ds_cfg['img_size']}×{ds_cfg['img_size']}"
        )

        if st.button("Start training"):
            run_training(dataset, n_epochs, batch_size, h, n_block, model_type)

    elif mode == "View results":
        st.header("Results")
        show_results()
        
    elif mode == "Image Completion":
        show_completion()


if __name__ == "__main__":
    main()