[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# Projet de Deep Learning

This repository serves as a template for the class "Projet de Deep Learning" in the 1st year of Master "Mathématiques et Intelligence Artificielle" of Université Paris-Saclay. Namely, this repository contains the code of a toy [Streamlit](https://streamlit.io/) application.

## Content

The user can select four modes in the sidebar of the application.
1. "Home data regression" performs regression with three simple methods (using decision trees and random forests) on a simple dataset of house prices. The user can select which covariates to use in the regression and visualize the validation MAE of the three methods.
2. "Sinus regression" performs regression with polynomial regression and decision trees on the sinus function, with noise. The user can select the density of noisy data points and the order of the polynoms. The regressors are then plotted, with the data points.
3. "Show MNIST" visualizes 6 random data points of the MNIST dataset and their labels.
4. "Deep Learning" trains (or use trained weights if available) a simple artificial neuron network on the Fashion MNIST datatset, displays the architecture, displays the curves of train and test loss and train and test accuracy, and finally visualizes 6 random data points of the dataset, their labels and the predicition of the model. The user can select the number of hidden layers (it is a simple MLP), the level of dropout and the number of epochs. If trained weights for the same combination of hidden layers and dropout are found, they are used. If not, a model is trained and then the weights and metrics are saved. In the former case, a button allows the user to delete the trained weights and metrics and start a new training.

## Miscellaneous

### pre-commit usage

This repo uses 2 pre-commit hooks: black and flake8. Contributors should install pre-commit (`pip install pre-commit`) and then run `pre-commit install` to install the hooks. Update the hooks with `pre-commit autoupdate`.

### docstrings

This repo uses the [numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html) for its docstrings.