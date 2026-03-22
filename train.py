import time
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image

from model import PixelCNN, PixelRNN, GatedPixelCNN
from Loader import get_loader, get_dataset_config

MODEL_REGISTRY = {
    "PixelCNN":      PixelCNN,
    "PixelRNN":      PixelRNN,
    "GatedPixelCNN": GatedPixelCNN,
}


def sample_pixels(
    model: nn.Module,
    canvas: torch.Tensor,
    row_start: int,
    row_end: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Generate pixels autoregressively for a given row range.

    Fills ``canvas[:, :, row_start:row_end, :]`` in-place, pixel by pixel,
    using the model's predicted distribution. The canvas must already
    contain any conditioning context (e.g. the top half of an image) in
    rows outside ``[row_start, row_end)``.

    Args:
        model (nn.Module): A trained autoregressive model in eval mode.
        canvas (torch.Tensor): Float tensor of shape (B, C, H, W) in [0, 1].
            Modified in-place and also returned for convenience.
        row_start (int): First row to generate (inclusive).
        row_end (int): Last row to generate (exclusive).
        temperature (float): Softmax temperature for sampling. Values < 1
            make the distribution sharper; values > 1 make it more uniform.

    Returns:
        torch.Tensor: The ``canvas`` tensor with the requested rows filled in.
    """
    device = canvas.device
    n_channel = canvas.shape[1]
    img_size = canvas.shape[2]

    model.eval()
    with torch.no_grad():
        for i in range(row_start, row_end):
            for j in range(img_size):
                logits_ij = model(canvas)[:, :, i, j, :]   # (B, C, 256)
                for ch in range(n_channel):
                    probs = F.softmax(logits_ij[:, ch, :] / temperature, dim=-1)
                    pixel = (
                        torch.multinomial(probs, 1).float() / 255.0
                    ).squeeze(-1)                            # (B,)
                    canvas[:, ch, i, j] = pixel

    return canvas


def build_data_loaders(
    dataset: str, batch_size: int, dataset_root: str = "./dataset"
) -> tuple:
    """
    Create train and test DataLoaders for the chosen dataset.

    Shared by the CLI entry point (``main.py``) and the Streamlit app
    (``app.py``) so the logic is never duplicated.

    Args:
        dataset (str): Dataset name (e.g. ``"CIFAR10"`` or ``"MNIST"``).
        batch_size (int): Number of images per mini-batch.
        dataset_root (str): Root directory where datasets are stored.

    Returns:
        tuple: A ``(train_loader, test_loader)`` tuple.
    """
    train_loader = get_loader(
        dataset_root, batch_size, train=True, dataset_name=dataset
    )
    test_loader = get_loader(
        dataset_root, batch_size, train=False, dataset_name=dataset
    )
    return train_loader, test_loader


class Solver(object):
    """
    Main class to manage training, evaluation and sampling
    procedures for PixelCNN models.
    """

    def __init__(self, config, train_loader, test_loader):
        """
        Initialise the Solver with data loaders and configuration.

        Args:
            config: BaseConfig instance with training hyper-parameters.
            train_loader: DataLoader for training data.
            test_loader: DataLoader for validation/test data.
        """
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_batches_per_epoch = len(self.train_loader)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.test_losses = []

    def build(self, model_override=None):
        """
        Build the model from configuration or use an external instance.

        Args:
            model_override (nn.Module, optional): External model instance to use
                instead of building one from config. Defaults to None.
        """
        if model_override is not None:
            self.model = model_override.to(self.device)
        else:
            ds_cfg = get_dataset_config(self.config.dataset)
            n_channel = ds_cfg["n_channel"]
            model_type = getattr(self.config, "model_type", "PixelCNN")
            model_class = MODEL_REGISTRY.get(model_type, PixelCNN)

            self.model = model_class(
                n_channel=n_channel,
                h=self.config.h,
                n_block=self.config.n_block,
            ).to(self.device)

        print(self.model, "\n")

        if self.config.mode == "train":
            self.optimizer = self.config.optimizer(
                self.model.parameters(),
                lr=getattr(self.config, "lr", 1e-3),
            )
            self.criterion = nn.CrossEntropyLoss()
            # Halve the lr when test loss has not improved for 3 epochs.
            # min_lr caps the decay so training never stalls completely.
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            )

    def train(self):
        """
        Run the training loop over all epochs and save model weights.

        At the end of each epoch the test loss is computed, the learning
        rate scheduler is stepped, and a sample image is saved to the
        checkpoint directory. Final model weights are saved as
        ``model_weights.pth``.
        """
        for epoch in trange(self.config.n_epochs, desc="Epoch", ncols=80):
            epoch += 1

            if epoch == 1:
                self.sample(epoch)

            self.model.train()
            batch_losses = []

            for batch_index, (images, labels) in enumerate(
                tqdm(self.train_loader, desc="Batch", ncols=80, leave=False)
            ):
                batch_index += 1
                images = images.to(self.device)
                logits = self.model(images)
                logits = logits.reshape(-1, 256)                 # (B*C*H*W, 256)
                targets = (images * 255).long().view(-1)         # (B*C*H*W,)

                loss = self.criterion(logits, targets)

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                loss_value = float(loss.detach())
                batch_losses.append(loss_value)

                if batch_index > 1 and batch_index % self.config.log_interval == 0:
                    tqdm.write(
                        f"Epoch: {epoch} | "
                        f"Batch: ({batch_index}/{self.num_batches_per_epoch})"
                        f" | Loss: {loss_value:.3f}"
                    )

            epoch_loss = float(np.mean(batch_losses))
            self.train_losses.append(epoch_loss)
            tqdm.write(f"Epoch Loss: {epoch_loss:.2f}")

            test_loss = self.test(epoch)
            self.test_losses.append(test_loss)

            self.scheduler.step(test_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            tqdm.write(f"LR: {current_lr:.2e}")

            self.sample(epoch)

        weights_path = str(self.config.ckpt_dir / "model_weights.pth")
        torch.save(self.model.state_dict(), weights_path)
        tqdm.write(f"Model saved to {weights_path}")

    def test(self, epoch) -> float:
        """
        Evaluate the model on the test set.

        Logs the mean cross-entropy loss in nats and, for interpretability,
        converts it to bits per dimension (bits/dim), which is the standard
        unit used in the autoregressive image modelling literature.

        Args:
            epoch (int): Current epoch number (used for logging).

        Returns:
            float: Mean test loss in nats.
        """
        test_losses = []
        start_time = time.time()
        self.model.eval()

        ds_cfg = get_dataset_config(self.config.dataset)
        n_channel = ds_cfg["n_channel"]

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                logits = self.model(images).reshape(-1, 256)
                targets = (images * 255).long().view(-1)
                loss = F.cross_entropy(logits, targets)
                test_losses.append(float(loss.detach()))

        test_time = time.time() - start_time
        mean_loss = float(np.mean(test_losses))
        bits_per_dim = mean_loss / np.log(2)
        tqdm.write(
            f"Test | Time: {test_time:.1f}s"
            f" | Loss: {mean_loss:.4f} nats"
            f" | {bits_per_dim:.4f} bits/dim"
            f" (channels: {n_channel})"
        )
        return mean_loss

    def sample(self, epoch) -> str:
        """
        Generate images pixel-by-pixel and save them to checkpoint directory.

        Args:
            epoch (int): Current epoch number (used for filename).

        Returns:
            str: Path to the saved image file.
        """
        image_path = str(self.config.ckpt_dir / f"epoch-{epoch}.png")
        tqdm.write(f"Saving sampled images at {image_path}")

        ds_cfg = get_dataset_config(self.config.dataset)
        n_channel = ds_cfg["n_channel"]
        img_size = ds_cfg["img_size"]
        n_samples = min(self.config.batch_size, 16)
        canvas = torch.zeros(
            n_samples, n_channel, img_size, img_size
        ).to(self.device)

        temperature = getattr(self.config, "sampling_temperature", 1.0)

        with trange(img_size, desc="Sampling Height", leave=False, ncols=80):
            sample_pixels(self.model, canvas, 0, img_size, temperature)

        save_image(canvas, image_path)
        return image_path
