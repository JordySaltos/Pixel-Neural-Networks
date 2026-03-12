"""This module contains the function used for visualization of an image dataset.

Notes
-----
Inspired by https://nextjournal.com/gkoehler/pytorch-mnist.
"""

import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np


def mnist_like_viz(data, classes, model=None):
    """Selects randomly 6 images from an MNIST-like dataset and plot them.

    Also plots the label as "GT" (ground truth) and, if a model is given,
    the prediction of the model on the given images.

    Parameters
    ----------
    data: torchvision.datasets.MNIST
        An MNIST-like dataset of grayscale labeled images.
    classes: list
        The list of the labale classes.
    model: torch.nn.module, default=None
        A torch model used to predict labels from the selected
        images. If None, there is simply no prediction.

    Returns
    -------
    None

    """
    if model is not None:
        model.eval()

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        sample_idx = torch.randint(len(data), size=(1,)).item()
        x, y = data[sample_idx][0], data[sample_idx][1]
        actual_class = classes[y]
        if model is not None:
            with torch.no_grad():
                pred = model(x)
                predicted = classes[pred[0].argmax(0)]
        plt.imshow(x[0], cmap="gray", interpolation="none")
        if model is None:
            plt.title("Ground Truth: {}".format(actual_class))
        else:
            plt.title(f"GT: {actual_class}, \nPred: {predicted}")
        plt.xticks([])
        plt.yticks([])
    st.pyplot(fig)


def training_curves(model, mode=None):
    """Plots metrics evolution during the training of a FMNIST_MLP model.

    Plots 2 figures that are the evolution
    of, respectively, the losses (train and test) and accuracies (train and test)
    with respect to the epoch.

    Parameters
    ----------
    model: dl.FMNIST_MLP
    mode: str
        Either `"script"` if the module is used as a script,
        or `"st"` if used in the stramlit app. This governs
        the kind of outputs produced (prints, figures).

    Returns
    -------
    None
    """

    fig = plt.figure()
    fig.set_figheight(10)
    padding = 2
    model.metrics.index = np.array(model.metrics.index) + 1
    plt.subplot(2, 1, 1)
    plt.tight_layout(pad=padding)
    plt.plot(model.metrics.index, "train_loss", data=model.metrics)
    plt.plot(model.metrics.index, "test_loss", data=model.metrics)
    plt.legend()
    plt.title("Train and test loss during training")
    plt.subplot(2, 1, 2)
    plt.tight_layout(pad=padding)
    plt.plot(model.metrics.index, "train_acc", data=model.metrics)
    plt.plot(model.metrics.index, "test_acc", data=model.metrics)
    plt.legend()
    plt.title("Train and test accuracy during training")
    if mode == "script":
        plt.show()
    elif mode == "st":
        st.pyplot(fig)
