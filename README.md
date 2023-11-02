# machine_learning_project2
Certainly, here's a README for the program:

# Using a Neural Network to Fit the California Housing Data

This program demonstrates the process of using a neural network to fit the California Housing data. It compares the performance of a neural network to that of a linear regression model. The California Housing dataset is used for this purpose.

## Data Preparation

- The program starts by loading the California Housing dataset from a remote source using urllib.
- It preprocesses the data, which includes removing the target variable ("median_house_value") and splitting the data into features (X) and target (y).
- A data transformation pipeline is created using scikit-learn's `Pipeline` and `ColumnTransformer`. This pipeline handles missing data imputation and feature scaling.

## Linear Regression Model

- The program uses scikit-learn's `LinearRegression` model to fit the data.
- It evaluates the model's performance on both training and testing data using the mean squared error.

## Neural Network with nn.Sequential()

- The program builds a neural network model using PyTorch's `nn.Sequential()` for fitting the data.
- It converts the data into PyTorch tensors.
- The neural network consists of two linear layers with a Tanh activation function and mean squared error loss function.
- The model is trained using stochastic gradient descent (SGD) for a specified number of epochs.

## Neural Network with Subclassing nn.Module

- A more flexible neural network model is created using PyTorch's `nn.Module` subclassing.
- The model architecture is defined with customizable hidden layers and activation functions.
- Training is implemented using a custom training loop for better control over the training process.

## Running the Program

To run this program, ensure you have the required Python libraries (scikit-learn and PyTorch) installed. You can execute the code to load the dataset, preprocess it, and fit both the linear regression and neural network models. The program evaluates and reports the performance of these models on the California Housing dataset.

Please note that this is a simplified example for educational purposes. In real-world scenarios, you may need to fine-tune hyperparameters and conduct more extensive testing and evaluation for a comprehensive analysis.
