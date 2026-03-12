"""The main module of the app.

Contains most of the functions governing the
different app modes.

"""

import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from viz import mnist_like_viz, training_curves
from utils import poly, paths
import dl


def main():
    """The main function of the app.

    Calls the appropriate mode function, depending on the user's choice
    in the sidebar. The mode function that can be called are
    `regression`, `sinus`, `mnist_viz`, and `fashionmnist`.

    Returns
    -------
    None
    """
    st.title("Some data manipulations")

    home_data = get_data()

    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            "Show instructions",
            "Home data regression",
            "Sinus regression",
            "Show MNIST",
            "Deep Learning",
        ],
    )  # , "Show the source code"])
    if app_mode == "Show instructions":
        st.write("To continue select a mode in the selection box to the left.")
    # elif app_mode == "Show the source code":
    #     st.code(get_file_content_as_string("./app.py"))
    elif app_mode == "Home data regression":
        regression(home_data)
    elif app_mode == "Sinus regression":
        sinus()
    elif app_mode == "Show MNIST":
        mnist()
    elif app_mode == "Deep Learning":
        fashionmnist()


@st.cache_data
def get_data():
    """Loads the home training data.

    Returns
    -------
    home_data: pd.DataFrame
        The home training data.

    Notes
    -----
    This is the dataset dowloaded from https://www.kaggle.com/competitions/home-data-for-ml-course/data.

    """
    iowa_file_path = "./home-data-for-ml-course/train.csv"
    home_data = pd.read_csv(iowa_file_path)
    return home_data


# def get_file_content_as_string(path):
#     with open(path) as f:
#         lines = f.read()
#     return lines


def regression(home_data):
    """Performs regression on the home training data.

    The dataset is split in a training and
    a validation sets.
    The user has the choice of which covariates to incoporate
    in the model. Then a decision tree, a decision tree
    with `max_leaf_nodes=100`, and a random forest are fitted
    on the training set. Finally the validation mean
    absolute errors are displayed.

    Parameters
    ----------
    home_data: pd.DataFrame
        The home training data. It can be any DataFrame except it needs
        the columns `SalePrice`, `LotArea`, `YearBuilt`, `1stFlrSF`,
        `2ndFlrSF`, `FullBath`, `BedroomAbvGr`, and `TotRmsAbvGrd`.

    Returns
    -------
    None

    """
    # Create target object and call it y
    y = home_data.SalePrice

    features = [
        "LotArea",
        "YearBuilt",
        "1stFlrSF",
        "2ndFlrSF",
        "FullBath",
        "BedroomAbvGr",
        "TotRmsAbvGrd",
    ]
    home_data_extracted = home_data[["SalePrice"] + features]

    st.text(
        "This is the head of the dataframe of Iowa house prices with many covariates"
    )
    st.write(home_data_extracted.head())

    # Create X
    covariates = st.multiselect(
        "Select covariates to keep for regression:", features, features
    )
    covariates.sort()
    X = home_data[covariates]

    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    dict_val_maes = {"method": [], "Val MAE": []}

    # Specify Model
    iowa_model = DecisionTreeRegressor(random_state=1)
    # Fit Model
    iowa_model.fit(train_X, train_y)
    # Make validation predictions and calculate mean absolute error
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    dict_val_maes["method"].append("DecisionTreeRegressor")
    dict_val_maes["Val MAE"].append(val_mae)

    # Using best value for max_leaf_nodes
    iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
    iowa_model.fit(train_X, train_y)
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    dict_val_maes["method"].append("DecisionTreeRegressor with max leaf nodes")
    dict_val_maes["Val MAE"].append(val_mae)

    # Define the model. Set random_state to 1
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    dict_val_maes["method"].append("RandomForestRegressor")
    dict_val_maes["Val MAE"].append(rf_val_mae)

    val_maes = pd.DataFrame(dict_val_maes).set_index("method")
    st.write(val_maes)
    st.text("(Test what happens when removing TotRmsAbvGrd)")


def sinus():
    """A simple example of regression on the sinus function on the interval [0,5].

    Some points are perturbed with noise after applying
    the sinus function to them.
    The user decides the number of noisy points with a slider,
    and the maximum order for the polynomial regression. They
    also decide if they want to fit two regression trees (with
    `max_depth=2` and `max_depth=5`) in addition to the polynomial
    regression. Then the fitted models are plotted along with
    the training noisy data.

    Returns
    -------
    None

    """
    noise = st.slider("Noise volume", 1, 10, 5, format="1 of each %d point(s)")
    # Order of the polynom for the linear regression with polynom
    order = st.slider(
        "Choose the order of the polynom for the polynomial regression", 2, 20, 3
    )
    trees = st.checkbox("Show decision trees", True)

    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1))
    y = np.sin(X).ravel()
    y[::noise] += 3 * (0.5 - rng.rand(y[::noise].size))
    X2 = poly(X, order=order)

    # Fit regression models
    if trees:
        regr_1 = DecisionTreeRegressor(max_depth=2, random_state=1)
        regr_2 = DecisionTreeRegressor(max_depth=5, random_state=1)
    regr_3 = LinearRegression()
    if trees:
        regr_1.fit(X, y)
        regr_2.fit(X, y)
    regr_3.fit(X2, y)

    # Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    X2_test = poly(X_test, order=order)
    if trees:
        y_1 = regr_1.predict(X_test)
        y_2 = regr_2.predict(X_test)
    y_3 = regr_3.predict(X2_test)

    # Plot the results
    fig = plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    if trees:
        plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
        plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.plot(X_test, y_3, color="red", label="polynom", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    if trees:
        plt.title("Decision Trees and Polynomial Regression")
    else:
        plt.title("Polynomial Regression")
    plt.xlim(-0.2, 5.2)
    plt.ylim(-2.7, 2.7)
    plt.legend()
    st.pyplot(fig)


def mnist():
    """Selects randomly 6 images from the training MNIST dataset and displays them.

    Returns
    -------
    None

    """
    train_data = MNIST("data", train=True, download=True, transform=ToTensor())
    classes = list(range(10))
    mnist_like_viz(train_data, classes)


def fashionmnist():
    """Training a simple MLP on the FashionMNIST dataset and displaying the metrics evolution during the training.

    The user can decide the number of hidden layers of the MLP. They can also choose the number of epochs
    for training. Once a model with given hyperparameters is trained, it is saved and used
    again the next times without new training, unless the user clicks the button to delete
    the saved model and train again. The MLP architecture is displayed.
    Then 2 figures that are the evolution
    of, respectively, the losses (train and test) and accuracies (train and test)
    with respect to the epoch, are displayed. Finally 6 random images of the test dataset are
    displayed, along with their ground truth and predicted labels.

    Returns
    -------
    None

    Notes
    -----
    Inspired by https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html.

    """
    st.header("A simple deep learning model applied on the FashionMNIST dataset")

    hidden_layers = st.slider("Choose the number of hidden layers", 1, 5, 2)

    dropout_rate = st.slider("Choose the dropout rate", 0.0, 0.9, 0.0, 0.1)

    epochs = st.slider("Choose the number of epochs to train", 1, 1000, 50)
    st.write(
        "Note that the epoch parameter is only relevant for training a new model, so if there is no already saved model for this config"
    )

    if st.button("Delete saved model and train again"):
        path_weights, path_metrics = paths(hidden_layers, dropout_rate)
        try:
            os.remove(path_weights)
            os.remove(path_metrics)
        except FileNotFoundError:
            pass

    train_dataloader, test_dataloader, _, test_data = dl.get_FashionMNIST_datasets(
        64, only_loader=False
    )
    model = dl.get_and_train_model(
        train_dataloader,
        test_dataloader,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        epochs=epochs,
        mode="st",
    )

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    training_curves(model, "st")
    mnist_like_viz(test_data, classes, model)


if __name__ == "__main__":
    main()
