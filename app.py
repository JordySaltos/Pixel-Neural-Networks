"""The main module of the app.

Contains most of the functions governing the different app modes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from Configuration import BaseConfig
from Loader import DATASET_CONFIGS, get_dataset_config, get_loader
from model import GatedPixelCNN, PixelCNN, PixelRNN
from train import Solver

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
WEIGHTS_FILENAME = "model_weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = "./dataset"

#: CIFAR-10 human-readable class names in label order.
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

#: Registry of available autoregressive models with metadata for the UI.
MODEL_CONFIGS = {
    "PixelCNN": {
        "class": PixelCNN,
        "description": (
            "Classic PixelCNN with masked convolutions and residual blocks. "
            "Fast to train, good baseline."
        ),
    },
    "PixelRNN": {
        "class": PixelRNN,
        "description": (
            "PixelRNN with Row-LSTM units. Captures long-range dependencies "
            "better than CNN but is significantly slower."
        ),
    },
    "GatedPixelCNN": {
        "class": GatedPixelCNN,
        "description": (
            "Gated PixelCNN with vertical + horizontal stacks. "
            "Eliminates the blind spot of the original PixelCNN."
        ),
    },
}

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def instantiate_model(
    model_type: str, n_channel: int, h: int, n_block: int
) -> torch.nn.Module:
    """
    Instantiate the requested model with the given architecture parameters.

    Each model class shares the same constructor signature
    (n_channel, h, n_block), so this function centralises the dispatch
    so callers never repeat the if/elif chain.

    Args:
        model_type (str): One of ``"PixelCNN"``, ``"PixelRNN"``,
            or ``"GatedPixelCNN"``.
        n_channel (int): Number of input/output image channels.
        h (int): Bottleneck / channel dimension.
        n_block (int): Number of residual or gated blocks.

    Returns:
        torch.nn.Module: An un-trained model placed on ``DEVICE``.

    Raises:
        ValueError: If ``model_type`` is not in ``MODEL_CONFIGS``.
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_type}'. Choose from {list(MODEL_CONFIGS)}."
        )
    model_class = MODEL_CONFIGS[model_type]["class"]
    return model_class(n_channel=n_channel, h=h, n_block=n_block).to(DEVICE)


@st.cache_resource
def load_model(
    weights_path: str,
    dataset: str,
    h: int,
    n_block: int,
    model_type: str = "PixelCNN",
) -> torch.nn.Module:
    """
    Load a trained model from disk and cache it across Streamlit reruns.

    Args:
        weights_path (str): Path to the ``.pth`` weights file.
        dataset (str): Dataset name used during training (e.g. ``"CIFAR10"``).
        h (int): Bottleneck dimension stored in the checkpoint config.
        n_block (int): Number of blocks stored in the checkpoint config.
        model_type (str): Architecture name stored in the checkpoint config.

    Returns:
        torch.nn.Module: The model in eval mode, placed on ``DEVICE``.
    """
    n_channel = get_dataset_config(dataset)["n_channel"]
    model = instantiate_model(model_type, n_channel, h, n_block)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Result-folder helpers
# ---------------------------------------------------------------------------


def list_result_folders() -> list:
    """
    Return all result folders that contain a weights file, newest first.

    Sorted by the timestamp embedded at the end of the folder name
    (``YYYY-MM-DD_HH-MM-SS``), so the order is purely chronological and
    unaffected by the model or dataset prefix. Falls back to filesystem
    modification time if the name does not contain a parseable timestamp.

    Returns:
        list: List of Path objects for valid result folders.
    """
    if not RESULTS_DIR.exists():
        return []

    def _folder_timestamp(folder: Path) -> str:
        """Extract the ``YYYY-MM-DD_HH-MM-SS`` suffix used as sort key."""
        parts = folder.name.split("_")
        if len(parts) >= 2:
            return "_".join(parts[-2:])
        return str(folder.stat().st_mtime)

    return sorted(
        (
            d
            for d in RESULTS_DIR.iterdir()
            if d.is_dir() and (d / WEIGHTS_FILENAME).exists()
        ),
        key=_folder_timestamp,
        reverse=True,
    )


def find_sample_images(folder: Path) -> list:
    """
    Return ``epoch-*.png`` checkpoint images from *folder*, sorted by epoch number.

    Uses numeric sort (epoch-10 comes after epoch-9) instead of the default
    lexicographic sort which would incorrectly place epoch-10 before epoch-9.

    Args:
        folder (Path): Path to the checkpoint directory.

    Returns:
        list: List of Path objects sorted by epoch number.
    """
    return sorted(
        folder.glob("epoch-*.png"),
        key=lambda p: int(p.stem.split("-")[1]),
    )


def read_config_from_folder(folder: Path) -> dict:
    """
    Parse the ``config.txt`` saved alongside a training run.

    Args:
        folder (Path): Path to the checkpoint directory.

    Returns:
        dict: Dictionary with keys ``dataset``, ``h``, ``n_block``, and
            ``model_type`` (whichever are present in the file).
    """
    config_file = folder / "config.txt"
    if not config_file.exists():
        return {}
    params = {}
    for line in config_file.read_text().splitlines():
        for key in ("dataset", "h", "n_block", "model_type"):
            if line.startswith(f"{key}:"):
                params[key] = line.split(":", 1)[1].strip()
    return params


def pick_run_folder(widget_key: str) -> "Path | None":
    """
    Render a selectbox for the user to choose a saved run.

    Args:
        widget_key (str): Unique Streamlit widget key (use the page name).

    Returns:
        Path | None: The selected folder as a ``Path``, or ``None`` if
            no runs exist.
    """
    folders = list_result_folders()
    if not folders:
        return None
    chosen = st.selectbox(
        "Select a saved run",
        [f.name for f in folders],
        index=0,
        key=f"run_selector_{widget_key}",
        help="Runs are listed newest first. Format: Model_Dataset_YYYY-MM-DD_HH-MM-SS",
    )
    return RESULTS_DIR / chosen


def load_run(widget_key: str) -> tuple:
    """
    Pick a run folder, read its config, and load the model.

    Combines ``pick_run_folder`` + ``read_config_from_folder`` + ``load_model``
    into a single call shared by all page functions.

    Args:
        widget_key (str): Forwarded to ``pick_run_folder`` as a unique widget key.

    Returns:
        tuple: ``(model, dataset, saved_config, folder)`` on success, or
            ``(None, None, None, None)`` when no runs are available.
            The folder is returned so callers can access checkpoint images
            without spawning a second selectbox widget.
    """
    folder = pick_run_folder(widget_key)
    if folder is None:
        st.warning("No trained model found. Go to **Train model** first.")
        return None, None, None, None

    saved = read_config_from_folder(folder)
    dataset = saved.get("dataset", "CIFAR10")
    h = int(saved.get("h", 128))
    n_block = int(saved.get("n_block", 15))
    model_type = saved.get("model_type", "PixelCNN")

    st.info(
        f"**{folder.name}** — "
        f"dataset: `{dataset}` | model: `{model_type}` | "
        f"h: `{h}` | n_block: `{n_block}`"
    )

    model = load_model(
        str(folder / WEIGHTS_FILENAME), dataset, h, n_block, model_type
    )
    return model, dataset, saved, folder


# ---------------------------------------------------------------------------
# Image rendering helpers
# ---------------------------------------------------------------------------


def tensor_to_numpy_grid(images: torch.Tensor, n_cols: int = 4) -> np.ndarray:
    """
    Convert a batch of image tensors to a numpy grid ready for imshow.

    Args:
        images (torch.Tensor): Float tensor of shape (B, C, H, W) in [0, 1].
        n_cols (int): Number of columns in the output grid.

    Returns:
        np.ndarray: Array of shape (H', W') for greyscale or (H', W', 3) for RGB.
    """
    grid = make_grid(images.cpu(), nrow=n_cols, normalize=True, pad_value=1)
    npimg = grid.numpy().transpose(1, 2, 0)
    if npimg.shape[2] == 1:
        npimg = npimg[:, :, 0]
    return np.clip(npimg, 0, 1)


def show_image_grid(
    images: torch.Tensor,
    title: str,
    cmap: "str | None",
    figsize: tuple,
    n_cols: int = 4,
) -> None:
    """
    Render a batch of tensors as a matplotlib grid inside Streamlit.

    Args:
        images (torch.Tensor): Float tensor of shape (B, C, H, W) in [0, 1].
        title (str): Figure title displayed above the grid.
        cmap (str | None): Matplotlib colormap (``"gray"`` for MNIST,
            ``None`` for RGB).
        figsize (tuple): ``(width, height)`` in inches.
        n_cols (int): Number of columns in the image grid.
    """
    npimg = tensor_to_numpy_grid(images, n_cols)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(npimg, cmap=cmap)
    ax.axis("off")
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_images(
    model: torch.nn.Module, dataset: str, n_images: int
) -> torch.Tensor:
    """
    Generate images unconditionally, pixel by pixel.

    Args:
        model (torch.nn.Module): A trained autoregressive model in eval mode.
        dataset (str): Dataset name, used to retrieve image dimensions.
        n_images (int): Number of images to generate.

    Returns:
        torch.Tensor: Float tensor of shape (n_images, C, H, W) in [0, 1].
    """
    ds_cfg = get_dataset_config(dataset)
    n_channel = ds_cfg["n_channel"]
    img_size = ds_cfg["img_size"]
    generated = torch.zeros(n_images, n_channel, img_size, img_size, device=DEVICE)

    model.eval()
    with torch.no_grad():
        for i in range(img_size):
            for j in range(img_size):
                probs = F.softmax(model(generated)[:, :, i, j], dim=2)
                for ch in range(n_channel):
                    pixel = (
                        torch.multinomial(probs[:, ch], 1).float() / 255.0
                    ).squeeze(-1)
                    generated[:, ch, i, j] = pixel
    return generated


def sample_conditional(
    model: torch.nn.Module,
    dataset: str,
    real_image: torch.Tensor,
    n_completions: int,
) -> torch.Tensor:
    """
    Complete the bottom half of an image from the top half context.

    Args:
        model (torch.nn.Module): A trained autoregressive model in eval mode.
        dataset (str): Dataset name, used to retrieve image dimensions.
        real_image (torch.Tensor): Single image tensor of shape
            (1, C, H, W) in [0, 1].
        n_completions (int): Number of independent completions to generate.

    Returns:
        torch.Tensor: Float tensor of shape (n_completions, C, H, W) in [0, 1].
    """
    ds_cfg = get_dataset_config(dataset)
    n_channel = ds_cfg["n_channel"]
    img_size = ds_cfg["img_size"]
    half = img_size // 2

    generated = real_image.repeat(n_completions, 1, 1, 1).to(DEVICE)
    generated[:, :, half:, :] = 0.0

    model.eval()
    with torch.no_grad():
        for i in range(half, img_size):
            for j in range(img_size):
                probs = F.softmax(model(generated)[:, :, i, j], dim=2)
                for ch in range(n_channel):
                    pixel = (
                        torch.multinomial(probs[:, ch], 1).float() / 255.0
                    ).squeeze(-1)
                    generated[:, ch, i, j] = pixel
    return generated


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _build_data_loaders(dataset: str, batch_size: int) -> tuple:
    """
    Create train and test DataLoaders for the chosen dataset.

    Args:
        dataset (str): Dataset name (e.g. ``"CIFAR10"`` or ``"MNIST"``).
        batch_size (int): Number of images per mini-batch.

    Returns:
        tuple: A ``(train_loader, test_loader)`` tuple.
    """
    train_loader = get_loader(
        DATASET_ROOT, batch_size, train=True, dataset_name=dataset
    )
    test_loader = get_loader(
        DATASET_ROOT, batch_size, train=False, dataset_name=dataset
    )
    return train_loader, test_loader


def _run_epoch(
    solver: Solver,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    n_epochs: int,
    progress_bar: st.delta_generator.DeltaGenerator,
    loss_placeholder: st.delta_generator.DeltaGenerator,
) -> None:
    """
    Execute one training epoch and update the Streamlit progress widgets.

    Args:
        solver (Solver): Initialised Solver with model, optimizer, and criterion.
        train_loader (DataLoader): DataLoader for the training split.
        epoch (int): Current epoch number (1-indexed).
        n_epochs (int): Total number of epochs (used for the progress bar fraction).
        progress_bar (DeltaGenerator): Streamlit progress widget to update.
        loss_placeholder (DeltaGenerator): Streamlit empty widget for live loss display.
    """
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

    test_loss = solver.test(epoch)
    solver.test_losses.append(test_loss)
    solver.sample(epoch)

    progress_bar.progress(epoch / n_epochs, text=f"Epoch {epoch}/{n_epochs}")
    loss_placeholder.write(
        f"**Epoch {epoch}** — train loss: `{epoch_loss:.4f}` | "
        f"test loss: `{test_loss:.4f}`"
    )


def _plot_losses(train_losses: list, test_losses: list) -> None:
    """
    Plot train and validation loss curves and display them in Streamlit.

    Args:
        train_losses (list): Mean training loss per epoch.
        test_losses (list): Mean validation loss per epoch.
    """
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(epochs, train_losses, marker="o", label="Train loss")
    ax.plot(epochs[: len(test_losses)], test_losses, marker="s", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def run_training(
    dataset: str,
    n_epochs: int,
    batch_size: int,
    h: int,
    n_block: int,
    model_type: str,
    lr: float,
) -> None:
    """
    Configure, build, and run a full training session from the Streamlit UI.

    Args:
        dataset (str): Dataset name (``"CIFAR10"`` or ``"MNIST"``).
        n_epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        h (int): Bottleneck / channel dimension.
        n_block (int): Number of residual / gated blocks.
        model_type (str): Architecture name (key in ``MODEL_CONFIGS``).
        lr (float): Initial learning rate.
    """
    config = BaseConfig().initialize(
        parse=False,
        mode="train",
        n_epochs=n_epochs,
        batch_size=batch_size,
        optimizer="Adam",
        lr=lr,
        dataset=dataset,
        h=h,
        n_block=n_block,
        model_type=model_type,
        log_interval=100,
        save_interval=10,
    )

    st.write(
        f"**Device:** `{DEVICE}` | **Dataset:** `{dataset}` | "
        f"**Model:** `{model_type}` | **h:** `{h}` | "
        f"**n_block:** `{n_block}` | **lr:** `{lr}`"
    )
    st.write(f"**Results folder:** `{config.ckpt_dir}`")

    train_loader, test_loader = _build_data_loaders(dataset, batch_size)
    n_channel = get_dataset_config(dataset)["n_channel"]
    solver_model = instantiate_model(model_type, n_channel, h, n_block)

    solver = Solver(config, train_loader, test_loader)
    solver.build(model_override=solver_model)

    if n_epochs > 0:
        solver.sample(1)

    progress_bar = st.progress(0, text="Starting training...")
    loss_placeholder = st.empty()

    for epoch in range(1, n_epochs + 1):
        _run_epoch(
            solver, train_loader, epoch, n_epochs, progress_bar, loss_placeholder
        )

    weights_path = str(config.ckpt_dir / WEIGHTS_FILENAME)
    torch.save(solver.model.state_dict(), weights_path)
    st.success(f"Training complete! Weights saved to `{weights_path}`")
    _plot_losses(solver.train_losses, solver.test_losses)


# ---------------------------------------------------------------------------
# Page: Image Generation
# ---------------------------------------------------------------------------


def _load_real_samples(dataset: str, n_images: int) -> torch.Tensor:
    """
    Load *n_images* real test-set images, cached in session_state.

    Args:
        dataset (str): Dataset name used to select the loader.
        n_images (int): Number of images to fetch.

    Returns:
        torch.Tensor: Float tensor of shape (n_images, C, H, W) in [0, 1].
    """
    cache_key = f"real_samples_{dataset}_{n_images}"
    if cache_key not in st.session_state:
        loader = get_loader(
            DATASET_ROOT, batch_size=n_images, train=False, dataset_name=dataset
        )
        imgs, _ = next(iter(loader))
        st.session_state[cache_key] = imgs
    return st.session_state[cache_key]


def show_generation() -> None:
    """
    Render the Image Generation page.

    Displays checkpoint samples from the selected training run and lets the
    user generate new images side-by-side with real dataset examples.
    """
    model, dataset, _, folder = load_run("generation")
    if model is None:
        return

    ds_cfg = get_dataset_config(dataset)
    cmap = "gray" if ds_cfg["n_channel"] == 1 else None

    sample_files = find_sample_images(folder)
    if sample_files:
        n_show = min(len(sample_files), 5)
        last_files = sample_files[-n_show:]
        st.subheader("Training progression")
        cols = st.columns(n_show)
        for col, img_path in zip(cols, last_files):
            col.image(
                str(img_path), caption=img_path.stem, use_container_width=True
            )

    st.subheader("Generate new images")
    n_images = st.slider("Number of images", 1, 16, 4)

    if st.button("Generate", type="primary"):
        fh = max(2, n_images // 4 * 2)
        col_real, col_gen = st.columns(2)

        real_imgs = _load_real_samples(dataset, n_images)
        with col_real:
            st.caption("Real images - dataset")
            show_image_grid(real_imgs, dataset, cmap, (4, fh))

        with st.spinner("Sampling pixel-by-pixel... this can take a while"):
            generated = sample_images(model, dataset, n_images)

        with col_gen:
            st.caption("Generated images - model")
            show_image_grid(generated, "model output", cmap, (4, fh))


# ---------------------------------------------------------------------------
# Page: Image Completion
# ---------------------------------------------------------------------------


def _load_test_dataset(dataset: str) -> tuple:
    """
    Load the full test set into memory, cached in session_state.

    Args:
        dataset (str): Dataset name.

    Returns:
        tuple: A ``(images, labels)`` tuple of tensors.
    """
    cache_key = f"test_images_{dataset}"
    if cache_key not in st.session_state:
        loader = get_loader(
            DATASET_ROOT, batch_size=10000, train=False, dataset_name=dataset
        )
        imgs, lbls = next(iter(loader))
        st.session_state[cache_key] = (imgs, lbls)
    return st.session_state[cache_key]


def _pick_image_index(
    dataset: str, test_labels: torch.Tensor, n_test: int
) -> tuple:
    """
    Render a class selector and return a stable (img_idx, class_label) pair.

    The index is persisted in ``st.session_state`` so it does not change
    when unrelated widgets trigger a Streamlit rerun. It only changes when
    the class selector value changes or the user clicks "Pick another image".

    Args:
        dataset (str): Controls which selector widget is shown (digit vs class).
        test_labels (torch.Tensor): Label tensor for the test set.
        n_test (int): Total number of test images (used for the random fallback).

    Returns:
        tuple: A ``(img_idx, class_label)`` tuple.
    """
    if dataset == "MNIST":
        chosen = st.selectbox("Select a digit (0-9)", list(range(10)))
        class_label = str(chosen)
        numeric_class = int(chosen)
        matching = (test_labels == chosen).nonzero(as_tuple=True)[0]
        if len(matching) == 0:
            st.warning(f"No images found for digit {chosen}.")
            st.stop()
    elif dataset == "CIFAR10":
        chosen = st.selectbox("Select a class", CIFAR10_CLASSES)
        numeric_class = CIFAR10_CLASSES.index(chosen)
        class_label = chosen
        matching = (test_labels == numeric_class).nonzero(as_tuple=True)[0]
        if len(matching) == 0:
            st.warning(f"No images found for class '{chosen}'.")
            st.stop()
    else:
        class_label = "random"
        numeric_class = -1
        matching = torch.arange(n_test)

    img_key = f"img_idx_{dataset}_{numeric_class}"
    prev_key = f"prev_class_{dataset}"

    if st.session_state.get(prev_key) != numeric_class or img_key not in st.session_state:
        st.session_state[img_key] = matching[
            torch.randint(len(matching), (1,)).item()
        ].item()
        st.session_state[prev_key] = numeric_class

    if st.button("Pick another image", key="resample_img"):
        st.session_state[img_key] = matching[
            torch.randint(len(matching), (1,)).item()
        ].item()
        st.rerun()

    return st.session_state[img_key], class_label


def show_completion() -> None:
    """
    Render the Image Completion page.

    The user picks a real test-set image; the model completes the masked
    bottom half pixel by pixel.
    """
    model, dataset, _, _folder = load_run("completion")
    if model is None:
        return

    st.write(
        "Select a real image from the test dataset. "
        "The model will complete the masked bottom half pixel by pixel."
    )

    ds_cfg = get_dataset_config(dataset)
    cmap = "gray" if ds_cfg["n_channel"] == 1 else None
    img_size = ds_cfg["img_size"]
    half = img_size // 2

    test_images, test_labels = _load_test_dataset(dataset)

    col1, col2 = st.columns([1, 2])
    with col1:
        img_idx, class_label = _pick_image_index(
            dataset, test_labels, len(test_images)
        )
    with col2:
        n_completions = st.slider("Number of variations", 1, 10, 1)

    real_image = test_images[img_idx: img_idx + 1].to(DEVICE)
    preview = real_image.cpu().squeeze(0).permute(1, 2, 0).numpy()
    preview = np.clip(preview, 0, 1)
    if preview.shape[2] == 1:
        preview = preview[:, :, 0]
    col1.image(
        preview,
        caption=f"Selected: {class_label}",
        use_container_width=True,
        clamp=True,
    )

    if st.button("Autocomplete bottom half", type="primary"):
        with st.spinner("Generating pixel by pixel... this may take a while"):
            completions = sample_conditional(
                model, dataset, real_image, n_completions
            )

        masked = real_image.clone()
        masked[:, :, half:, :] = 0.0
        all_images = torch.cat(
            [masked.cpu(), completions.cpu(), real_image.cpu()], dim=0
        )
        title = (
            f"Masked  |  {n_completions} "
            f"completion{'s' if n_completions > 1 else ''}  |  Original"
        )
        show_image_grid(
            all_images,
            title,
            cmap,
            (max(8.0, (2 + n_completions) * 1.5), 3.5),
            n_cols=2 + n_completions,
        )


# ---------------------------------------------------------------------------
# Page: Camera Completion
# ---------------------------------------------------------------------------


def _preprocess_camera_image(img_array: np.ndarray) -> torch.Tensor:
    """
    Convert a raw camera numpy array to a 32x32 CIFAR-style tensor.

    Drops the alpha channel if present (RGBA -> RGB), resizes to 32x32
    using Lanczos interpolation, and normalizes to [0, 1].

    Args:
        img_array (np.ndarray): uint8 array of shape (H, W, 3) or (H, W, 4).

    Returns:
        torch.Tensor: Float tensor of shape (1, 3, 32, 32) in [0, 1].
    """
    from PIL import Image as PILImage

    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    pil_img = PILImage.fromarray(img_array.astype(np.uint8))
    pil_img = pil_img.resize((32, 32), PILImage.LANCZOS)
    arr = np.array(pil_img).astype(np.float32) / 255.0           # (32, 32, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 32, 32)
    return tensor


def _show_pixel_grid(tensor: torch.Tensor, title: str, zoom: int = 8) -> None:
    """
    Display a 32x32 tensor as a zoomed pixel grid in Streamlit.

    Args:
        tensor (torch.Tensor): Float tensor of shape (1, 3, 32, 32) in [0, 1].
        title (str): Caption displayed below the image.
        zoom (int): Upscale factor for display (default 8, shown as 256x256).
    """
    from PIL import Image as PILImage

    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    pil_img = PILImage.fromarray(arr).resize(
        (32 * zoom, 32 * zoom), PILImage.NEAREST
    )
    st.image(pil_img, caption=title, use_container_width=False)


def show_camera_completion() -> None:
    """
    Render the Camera Completion page.

    The user takes a photo with their webcam. The app resizes it to 32x32,
    erases the bottom half, and uses the trained CIFAR-10 model to
    autocomplete the missing pixels.
    """
    st.write(
        "Take a photo with your webcam. It will be resized to **32x32 pixels** "
        "(CIFAR-10 format). The bottom half will be erased and the model will "
        "complete it pixel by pixel."
    )

    model, dataset, _, _folder = load_run("camera")
    if model is None:
        return

    if dataset != "CIFAR10":
        st.warning(
            f"The selected run was trained on **{dataset}**. "
            "Camera completion works best with a **CIFAR10** model (3-channel RGB)."
        )

    st.subheader("Step 1 - Take a photo")
    camera_image = st.camera_input("Click the button to capture")

    if camera_image is None:
        st.info("Waiting for a photo...")
        return

    from PIL import Image as PILImage
    import io
    pil_raw = PILImage.open(io.BytesIO(camera_image.getvalue())).convert("RGB")
    img_array = np.array(pil_raw)

    st.subheader("Step 2 - Image at 32x32 pixels")
    img_tensor = _preprocess_camera_image(img_array)   # (1, 3, 32, 32)

    half = 16
    masked = img_tensor.clone()
    masked[:, :, half:, :] = 0.0

    col_orig, col_masked = st.columns(2)
    with col_orig:
        _show_pixel_grid(img_tensor, "Original resized to 32x32")
    with col_masked:
        _show_pixel_grid(masked, "Bottom half erased (model input)")

    st.subheader("Step 3 - Model autocomplete")
    n_completions = st.slider("Number of variations", 1, 6, 3, key="cam_completions")

    if st.button("Autocomplete bottom half", type="primary", key="cam_run"):
        img_tensor = img_tensor.to(DEVICE)

        with st.spinner("Generating pixel by pixel... this may take a while"):
            completions = sample_conditional(
                model, dataset, img_tensor, n_completions
            )  # (n_completions, 3, 32, 32)

        st.success("Done! Here are the completions:")

        cols = st.columns(2 + n_completions)

        with cols[0]:
            _show_pixel_grid(masked, "Masked input")

        for k in range(n_completions):
            with cols[k + 1]:
                _show_pixel_grid(
                    completions[k: k + 1].cpu(),
                    f"Completion {k + 1}",
                )

        with cols[-1]:
            _show_pixel_grid(img_tensor.cpu(), "Original")


# ---------------------------------------------------------------------------
# Sidebar and page routing
# ---------------------------------------------------------------------------


def _sidebar() -> str:
    """
    Render the sidebar and return the selected mode name.

    Returns:
        str: One of ``"Train model"``, ``"Image Generation"``,
            ``"Image Completion"``, or ``"Camera Completion"``.
    """
    with st.sidebar:
        st.title("Pixel Neural Networks")
        st.caption("Autoregressive image generation")
        st.divider()
        mode = st.radio(
            "Mode",
            [
                "Train model",
                "Image Generation",
                "Image Completion",
                "Camera Completion",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"Device: `{DEVICE}`")
    return mode


def _page_train() -> None:
    """Render the Train model page with all configuration widgets."""
    st.header("Train a Pixel Neural Network")

    st.subheader("Data & training")
    col1, col2, col3 = st.columns(3)
    with col1:
        dataset = st.selectbox("Dataset", list(DATASET_CONFIGS.keys()))
    with col2:
        n_epochs = st.slider("Epochs", 1, 150, 20)
    with col3:
        batch_size = st.selectbox("Batch size", [8, 16, 32, 64, 128], index=2)

    st.subheader("Architecture")
    model_type = st.selectbox(
        "Model",
        list(MODEL_CONFIGS.keys()),
        help="Choose the autoregressive architecture to train.",
    )
    st.caption(MODEL_CONFIGS[model_type]["description"])

    col4, col5 = st.columns(2)
    with col4:
        h = st.slider(
            "h - channel dimension",
            16,
            256,
            128,
            step=16,
            help=(
                "PixelCNN / PixelRNN: bottleneck width of each residual block. "
                "GatedPixelCNN: number of channels throughout the network."
            ),
        )
    with col5:
        n_block = st.slider(
            "n_block - depth",
            4,
            20,
            10,
            step=1,
            help="Number of residual / gated blocks. More = larger receptive field.",
        )

    st.subheader("Optimizer")
    lr = st.select_slider(
        "Learning rate",
        options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2],
        value=1e-3,
        format_func=lambda x: f"{x:.0e}",
        help="Adam learning rate. Start with 1e-3; reduce if loss oscillates.",
    )

    ds_cfg = get_dataset_config(dataset)
    st.caption(
        f"Channels: {ds_cfg['n_channel']} | "
        f"Image size: {ds_cfg['img_size']}x{ds_cfg['img_size']}"
    )

    if st.button("Start training", type="primary", use_container_width=True):
        run_training(dataset, n_epochs, batch_size, h, n_block, model_type, lr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Configure the Streamlit page and dispatch to the selected mode."""
    st.set_page_config(
        page_title="Pixel Neural Networks",
        layout="wide",
    )
    mode = _sidebar()

    if mode == "Train model":
        _page_train()
    elif mode == "Image Generation":
        st.header("Image Generation")
        show_generation()
    elif mode == "Image Completion":
        st.header("Image Completion")
        show_completion()
    elif mode == "Camera Completion":
        st.header("Camera Completion")
        show_camera_completion()


if __name__ == "__main__":
    main()
