# kLogistics

This repository contains the implementation of various classifiers designed to solve binary and multiclass classification problems. The models are built using Python and NumPy. Below is an overview of the key classes and their functionalities:

### Classifier (Base Class)
- **`Classifier`**: The base class that defines the structure for classifiers.
  - Methods:
    - `fit(X, t)`: Initializes the classifier with the dataset.
    - `predict(X)`: Placeholder for prediction, implemented in subclasses.
    - `exp(x)`: Computes the exponential of a value.
    - `get_w()`, `set_w(w)`: Getter and setter for weights.
    - `get_eta(ii)`: Computes the learning rate based on iteration.

### Binary Classifier
- **`Binary`**: A binary classifier that predicts two possible outcomes.
  - Methods:
    - `sigma(x)`: Sigmoid activation function.
    - `predict(X)`: Predicts binary class labels.
    - `predictProb(X)`: Predicts probabilities for binary outcomes.

### Multiclass Classifier
- **`Multi`**: A multiclass classifier capable of handling more than two classes.
  - Methods:
    - `get_T(t)`: Converts labels to a one-hot encoded matrix.
    - `softmax(phi_n)`: Softmax function for multiclass prediction.
    - `predict(X)`: Predicts class labels for multiclass classification.
    - `predictProb(X)`: Predicts class probabilities.

### Logistic Binary Classifier
- **`LogisticBinary`**: Extends the binary classifier using logistic regression.
  - Methods:
    - `fit(X, t, verbose=0)`: Fits the binary logistic regression model to the data.

### K-Logistic Binary Classifier
- **`KLogisticBinary`**: A variant of the logistic binary classifier using a k-logistic function.
  - Methods:
    - `fit(X, t, verbose=0)`: Fits the k-logistic binary regression model.

### K-Logistic Multiclass Classifier
- **`KLogisticMulti`**: A multiclass classifier using the k-logistic function.
  - Methods:
    - `fit(X, t, verbose=0)`: Fits the k-logistic multiclass model to the data.

### Logistic Multiclass Classifier
- **`LogisticMulti`**: A multiclass logistic regression classifier.
  - Methods:
    - `fit(X, t, verbose=0)`: Fits the logistic multiclass model to the data.

### K Class
- **`K`**: A helper class to manage the parameter `k` used in k-logistic functions.
  - Methods:
    - `exp(x)`: A special function that uses the `k` parameter.

All classifiers are customizable with parameters like learning rate (`eta`), number of iterations (`MAXITER`), and more. These implementations allow for flexibility in both binary and multiclass classification tasks, making them suitable for a variety of datasets.

