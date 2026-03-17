import time
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from model import PixelCNN, PixelRNN, GatedPixelCNN
from Loader import get_dataset_config

MODEL_REGISTRY = {
    "PixelCNN":      PixelCNN,
    "PixelRNN":      PixelRNN,
    "GatedPixelCNN": GatedPixelCNN,
}


class Solver(object):
    """
    Main class that manages the training, evaluation and sampling
    procedures for the PixelCNN model.
    """

    def __init__(self, config, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.num_batches_per_epoch = len(self.train_loader)
        self.dataset_size = len(self.train_loader.dataset)

        self.is_train = self.config.isTrain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_losses = []
        self.test_losses = []

    def build(self, model_override=None):
        """
        Builds the model using architecture params from config,
        or uses the externally supplied `model_override` instance.
        Then initializes optimizer and loss function.

        Args:
            model_override: an already-instantiated nn.Module to use instead
                            of the default PixelCNN. Used by app.py when the
                            user selects PixelRNN or GatedPixelCNN.
        """
        if model_override is not None:
            self.model = model_override.to(self.device)
        else:
            # Default: build from config (used by main.py / CLI)
            ds_cfg = get_dataset_config(self.config.dataset)
            n_channel = ds_cfg["n_channel"]
            model_type = getattr(self.config, "model_type", "PixelCNN")
            model_class = MODEL_REGISTRY.get(model_type, PixelCNN)

            if model_type == "PixelRNN":
                self.model = model_class(
                    n_channel=n_channel, h=h, n_block=n_block
                ).to(self.device)
            elif model_type == "GatedPixelCNN":
                self.model = model_class(
                    in_channels=n_channel,
                    channels=self.config.h,
                    n_layers=self.config.n_block,
                ).to(self.device)
            else:
                self.model = model_class(
                    n_channel=n_channel,
                    h=self.config.h,
                    n_block=self.config.n_block,
                ).to(self.device)

        print(self.model, "\n")

        if self.config.mode == "train":
            self.optimizer = self.config.optimizer(self.model.parameters())
            self.criterion = nn.CrossEntropyLoss()

    def train(self):
        """
        Runs the full training loop across all epochs.
        Saves model weights to the checkpoint directory at the end.
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
                logits = logits.contiguous().view(-1, 256)
                targets = (images.view(-1) * 255).long()

                loss = self.criterion(logits, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_value = float(loss.detach())
                batch_losses.append(loss_value)

                if batch_index > 1 and batch_index % self.config.log_interval == 0:
                    tqdm.write(
                        f"Epoch: {epoch} | Batch: ({batch_index}/{self.num_batches_per_epoch})"
                        f" | Loss: {loss_value:.3f}"
                    )

            epoch_loss = float(np.mean(batch_losses))
            self.train_losses.append(epoch_loss)
            tqdm.write(f"Epoch Loss: {epoch_loss:.2f}")

            test_loss = self.test(epoch)
            self.test_losses.append(test_loss)

            self.sample(epoch)

        weights_path = str(self.config.ckpt_dir / "model_weights.pth")
        torch.save(self.model.state_dict(), weights_path)
        tqdm.write(f"Model saved to {weights_path}")

    def test(self, epoch):
        """Evaluates the model on the test set. Returns mean test loss."""
        test_losses = []
        start_time = time.time()
        self.model.eval()

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                logits = self.model(images).contiguous().view(-1, 256)
                targets = (images.view(-1) * 255).long()
                loss = F.cross_entropy(logits, targets)
                test_losses.append(float(loss.detach()))

        test_time = time.time() - start_time
        mean_loss = float(np.mean(test_losses))
        tqdm.write(
            f"Test completed | Time: {test_time:.1f}s | Test Loss: {mean_loss:.2f}"
        )
        return mean_loss

    def sample(self, epoch):
        """
        Generates images pixel-by-pixel and saves them to the checkpoint dir.
        Returns the path of the saved image.
        """
        image_path = str(self.config.ckpt_dir / f"epoch-{epoch}.png")
        tqdm.write(f"Saving sampled images at {image_path}")

        self.model.eval()

        ds_cfg = get_dataset_config(self.config.dataset)
        n_channel = ds_cfg["n_channel"]
        img_size = ds_cfg["img_size"]   # always 32 after padding

        generated_images = torch.zeros(
            self.config.batch_size, n_channel, img_size, img_size
        ).to(self.device)

        with torch.no_grad():
            for i in trange(img_size, desc="Sampling Height", leave=False, ncols=80):
                for j in range(img_size):
                    output = self.model(generated_images)
                    probabilities = F.softmax(output[:, :, i, j], dim=2)
                    for channel in range(n_channel):
                        sampled_pixel = (
                            torch.multinomial(probabilities[:, channel], 1).float() / 255.0
                        ).squeeze(-1)
                        generated_images[:, channel, i, j] = sampled_pixel

        save_image(generated_images, image_path)
        return image_path