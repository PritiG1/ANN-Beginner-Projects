# ANN-Beginner-Projects
A collection of basic artificial neural network (ANN) training examples for classification and regression problems, providing a starting point for understanding and implementing ANN models.

## Folder: Regression-Exercise

### Dataset

The dataset used in this exercise is obtained from the UCI Machine Learning Repository. It is stored in the file `Folds5x2_pp.xlsx`. The dataset contains a number of features and a target variable. 

### Requirements

To run the code in this repository, you need to have the following dependencies:

- Python (version 3.6 or higher)
- Numpy (version 1.19.5 or higher)
- Pandas (version 1.3.1 or higher)
- TensorFlow (version 2.6.0 or higher)
- Scikit-learn (version 0.24.2 or higher)

You can install these dependencies using pip:

```
pip install numpy pandas tensorflow scikit-learn
```

The code is divided into three main parts:

### 1. Data Preprocessing

In this part, the dataset is imported and preprocessed. The dataset is read from the `Folds5x2_pp.xlsx` file using the pandas library. The features (X) and target variable (y) are extracted from the dataset.

The dataset is then split into training and test sets using the `train_test_split` function from scikit-learn.

### 2. Building the ANN

This part involves building the Artificial Neural Network model using TensorFlow. The model is initialized using the `Sequential` class from `tf.keras.models`. 

Layers are added to the model using the `add` method. Two hidden layers with ReLU activation functions are added, and an output layer is added with a linear activation function.

### 3. Training the ANN

In this part, the ANN model is compiled and trained on the training set. The model is compiled with the Adam optimizer and mean squared error loss function using the `compile` method. The model is then trained on the training set using the `fit` method, specifying the batch size and number of epochs.

### 4. Evaluating the Model

After training the model, predictions are made on the test set using the `predict` method. The `r2_score` function from scikit-learn is used to calculate the R^2 score, which measures the performance of the model. The R^2 score is printed to evaluate the accuracy of the predictions.

Feel free to explore and modify the code as per your needs.

## Folder: Classification-Exercise


### Dataset

The dataset used in this example is the "Churn_Modelling" dataset, which is stored in the file `Churn_Modelling.csv`. The dataset contains information about customers, such as their credit score, gender, age, tenure, balance, number of products, and more. The target variable indicates whether the customer has churned or not.


The code is divided into four main parts:

### 1. Data Preprocessing

In this part, the dataset is imported and preprocessed. The dataset is read from the `Churn_Modelling.csv` file using the pandas library. The features (X) and the target variable (y) are extracted from the dataset.

The categorical feature 'Gender' is label encoded using the `LabelEncoder` from scikit-learn. The 'Geography' feature is one-hot encoded using the `ColumnTransformer` and `OneHotEncoder` from scikit-learn.

The dataset is then split into training and test sets using the `train_test_split` function from scikit-learn. Additionally, feature scaling is applied to standardize the feature values.

### 2. Building the ANN

In this part, the Artificial Neural Network (ANN) model is built using TensorFlow. The model is initialized using the `Sequential` class from `tf.keras.models`.

The input layer, two hidden layers, and the output layer are added using the `add` method. The hidden layers use the ReLU activation function, while the output layer uses the sigmoid activation function for binary classification.

### 3. Training the ANN

The ANN model is compiled with the Adam optimizer and the binary cross-entropy loss function using the `compile` method. The model is then trained on the training set using the `fit` method, specifying the batch size and the number of epochs.

### 4. Making Predictions and Evaluating the Model

After training the model, predictions are made on the test set using the `predict` method. The predictions are thresholded at 0.5 to obtain binary predictions.

The confusion matrix and the accuracy score are calculated using the `confusion_matrix` and `accuracy_score` functions from scikit-learn. These metrics provide insights into the performance of the model.

A single observation is also provided as an example to demonstrate how to make predictions on new data using the trained model.

Feel free to explore and modify the code as per your needs.

## Acknowledgments

This code is based on a classification example from a free Udemy course on artificial neural networks. The dataset used in this example is sourced from the UCI Machine Learning Repository.

## References

- UCI Machine Learning Repository: [https://archive.ics.uci.edu](https://archive.ics.uci.edu)
- TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Scikit-learn: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
