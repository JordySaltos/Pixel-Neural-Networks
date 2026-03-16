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
from torchvision.utils import make_grid, save_image

from Configuration import BaseConfig
from Loader import get_loader
from model import PixelCNN
from train import Solver


# ── paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
WEIGHTS_FILENAME = "model_weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── helpers ───────────────────────────────────────────────────────────────────

def find_latest_weights() -> Path | None:
    """Return the path to model_weights.pth in the most recent results folder."""
    if not RESULTS_DIR.exists():
        return None
    candidates = sorted(
        [d / WEIGHTS_FILENAME for d in RESULTS_DIR.iterdir()
         if d.is_dir() and (d / WEIGHTS_FILENAME).exists()]
    )
    return candidates[-1] if candidates else None


def find_sample_images() -> list[Path]:
    """Return all epoch-*.png files from the most recent results folder, sorted."""
    if not RESULTS_DIR.exists():
        return []
    folders = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()])
    if not folders:
        return []
    latest = folders[-1]
    return sorted(latest.glob("epoch-*.png"))


@st.cache_resource
def load_model(weights_path: str) -> PixelCNN:
    model = PixelCNN().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


# ── training ──────────────────────────────────────────────────────────────────

def run_training(n_epochs: int, batch_size: int):
    """Runs Solver training inside the Streamlit app."""

    config = BaseConfig().initialize(
        parse=False,
        mode="train",
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer="RMSprop",
        dataset="CIFAR10",
        log_interval=100,
        save_interval=10,
    )

    st.write(f"**Device:** {DEVICE}  |  **Results folder:** `{config.ckpt_dir}`")

    train_loader = get_loader(config.dataset_dir, batch_size, train=True)
    test_loader  = get_loader(config.dataset_dir, batch_size, train=False)

    solver = Solver(config, train_loader, test_loader)
    solver.build()

    progress = st.progress(0, text="Starting training…")
    loss_placeholder = st.empty()

    for epoch in range(1, n_epochs + 1):

        if epoch == 1:
            solver.sample(epoch)

        # ── train one epoch ──────────────────────────────────────────────
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

        # ── test one epoch ───────────────────────────────────────────────
        test_loss = solver.test(epoch)
        solver.test_losses.append(test_loss)

        # ── sample ───────────────────────────────────────────────────────
        img_path = solver.sample(epoch)

        # ── update UI ────────────────────────────────────────────────────
        progress.progress(epoch / n_epochs, text=f"Epoch {epoch}/{n_epochs}")
        loss_placeholder.write(
            f"**Epoch {epoch}** — train loss: `{epoch_loss:.4f}` | "
            f"test loss: `{test_loss:.4f}`"
        )

    # save weights
    weights_path = str(config.ckpt_dir / WEIGHTS_FILENAME)
    torch.save(solver.model.state_dict(), weights_path)
    st.success(f"Training complete! Weights saved to `{weights_path}`")

    # plot loss curves
    _plot_losses(solver.train_losses, solver.test_losses)

    return solver


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


# ── sampling / results ────────────────────────────────────────────────────────

def show_results():
    """Shows generated images saved during training and lets user generate new ones."""

    weights_path = find_latest_weights()

    if weights_path is None:
        st.warning("No trained model found. Go to **Train model** first.")
        return

    st.write(f"Loaded weights from `{weights_path}`")
    model = load_model(str(weights_path))

    # ── images saved during training ─────────────────────────────────────
    sample_files = find_sample_images()
    if sample_files:
        st.subheader("Images generated during training")
        cols = st.columns(min(len(sample_files), 5))
        for col, img_path in zip(cols, sample_files[-5:]):   # last 5 epochs
            col.image(str(img_path), caption=img_path.stem, use_container_width=True)

    # ── generate new images on demand ────────────────────────────────────
    st.subheader("Generate new images")
    n_images = st.slider("Number of images to generate", 1, 16, 4)

    if st.button("Generate"):
        with st.spinner("Sampling pixel-by-pixel… (this can take a while)"):
            generated = _sample(model, n_images)

        grid = make_grid(generated.cpu(), nrow=4, normalize=True, pad_value=1)
        npimg = grid.numpy().transpose(1, 2, 0)

        fig, ax = plt.subplots(figsize=(8, max(2, n_images // 4 * 2)))
        ax.imshow(np.clip(npimg, 0, 1))
        ax.axis("off")
        ax.set_title("PixelCNN generated samples")
        st.pyplot(fig)


def _sample(model: PixelCNN, n_images: int) -> torch.Tensor:
    model.eval()
    generated = torch.zeros(n_images, 3, 32, 32, device=DEVICE)

    with torch.no_grad():
        for i in range(32):
            for j in range(32):
                output = model(generated)                   # [B, C, H, W, 256]
                probs  = F.softmax(output[:, :, i, j], dim=2)  # [B, C, 256]
                for ch in range(3):
                    pixel = (
                        torch.multinomial(probs[:, ch], 1).float() / 255.0
                    ).squeeze(-1)
                    generated[:, ch, i, j] = pixel

    return generated


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("PixelCNN — CIFAR-10 image generation")

    mode = st.sidebar.selectbox(
        "Choose mode",
        ["Train model", "View results"],
    )

    if mode == "Train model":
        st.header("Train a PixelCNN")
        n_epochs   = st.slider("Number of epochs", 1, 50, 5)
        batch_size = st.selectbox("Batch size", [4, 8, 16, 32], index=1)

        if st.button("Start training"):
            run_training(n_epochs, batch_size)

    elif mode == "View results":
        st.header("Results")
        show_results()


if __name__ == "__main__":
    main()