import time
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


class Solver(object):

    """
    Main class that manages the training, evaluation and sampling
    procedures for the PixelCNN model.
    """

    def __init__(self, config, train_loader, test_loader):
        """
        Initializes the solver with configuration parameters and datasets.
        Stores training/test loaders and prepares device selection.
        """

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.num_batches_per_epoch = len(self.train_loader)
        self.dataset_size = len(self.train_loader.dataset)

        self.is_train = self.config.isTrain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def build(self):
        """
        Builds the PixelCNN model and initializes optimizer and loss function.
        """

        self.model = PixelCNN().to(self.device)
        print(self.model, "\n")

        if self.config.mode == "train":
            self.optimizer = self.config.optimizer(self.model.parameters())
            self.criterion = nn.CrossEntropyLoss()


    def train(self):
        """
        Runs the full training loop across all epochs.
        For each epoch it trains the network, evaluates it on the test set,
        and generates sample images.
        """

        for epoch in trange(self.config.n_epochs, desc="Epoch", ncols=80):
            epoch += 1

            # Debug sampling at first epoch
            if epoch == 1:
                self.sample(epoch)

            self.model.train()
            batch_losses = []

            for batch_index, (images, labels) in enumerate(
                tqdm(self.train_loader, desc="Batch", ncols=80, leave=False)
            ):

                batch_index += 1
                images = images.to(self.device)

                # Forward pass
                logits = self.model(images)
                logits = logits.contiguous().view(-1, 256)

                # Convert pixel values into class labels (0-255)
                targets = (images.view(-1) * 255).long()

                loss = self.criterion(logits, targets)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_value = float(loss.detach())
                batch_losses.append(loss_value)

                if batch_index > 1 and batch_index % self.config.log_interval == 0:
                    log_msg = (
                        f"Epoch: {epoch} | Batch: ({batch_index}/{self.num_batches_per_epoch}) "
                        f"| Loss: {loss_value:.3f}"
                    )
                    tqdm.write(log_msg)

            epoch_loss = np.mean(batch_losses)
            tqdm.write(f"Epoch Loss: {epoch_loss:.2f}")

            self.test(epoch)
            self.sample(epoch)


    def test(self, epoch):
        """
        Evaluates the trained model on the test dataset and reports
        the average cross-entropy loss.
        """

        test_losses = []
        start_time = time.time()

        self.model.eval()

        with torch.no_grad():

            for images, labels in self.test_loader:

                images = images.to(self.device)

                logits = self.model(images).contiguous()
                logits = logits.view(-1, 256)

                targets = (images.view(-1) * 255).long()

                loss = F.cross_entropy(logits, targets)

                test_losses.append(float(loss.detach()))

        test_time = time.time() - start_time

        log_msg = (
            f"Test completed | Time: {test_time:.1f}s | "
            f"Test Loss: {np.mean(test_losses):.2f}"
        )

        tqdm.write(log_msg)


    def sample(self, epoch):
        """
        Generates new images pixel-by-pixel using the autoregressive
        property of PixelCNN and saves them to disk.
        """

        image_path = str(self.config.ckpt_dir.joinpath(f"epoch-{epoch}.png"))
        tqdm.write(f"Saving sampled images at {image_path}")

        self.model.eval()

        # Empty image that will be progressively filled
        generated_images = torch.zeros(
            self.config.batch_size, 3, 32, 32
        ).to(self.device)

        with torch.no_grad():

            for i in trange(32, desc="Sampling Height", leave=False, ncols=80):
                for j in trange(32, desc="Sampling Width", leave=False, ncols=80):

                    output = self.model(generated_images)

                    probabilities = F.softmax(output[:, :, i, j], dim=2)

                    # Sample each channel independently
                    for channel in range(3):

                        sampled_pixel = (
                            torch.multinomial(probabilities[:, channel], 1).float() / 255.0
                        ).squeeze(-1)

                        generated_images[:, channel, i, j] = sampled_pixel

        save_image(generated_images, image_path)