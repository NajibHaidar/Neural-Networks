# Neural-Networks

### Table of Contents
[Abstract](#Abstract)
<a name="Abstract"/>

[Sec. I. Introduction and Overview](#sec-i-introduction-and-overview)     
<a name="sec-i-introduction-and-overview"/>

[Sec. II. Theoretical Background](#sec-ii-theoretical-background)     
<a name="sec-ii-theoretical-background"/>

[Sec. III. Algorithm Implementation and Development](#sec-iii-algorithm-implementation-and-development)
<a name="sec-iii-algorithm-implementation-and-development"/>

[Sec. IV. Computational Results](#sec-iv-computational-results)
<a name="sec-iv-computational-results"/>

[Sec. V. Summary and Conclusions](#sec-v-summary-and-conclusions)
<a name="sec-v-summary-and-conclusions"/>

### Abstract

This document explores the application of machine learning models, with a particular focus on neural networks, to model and predict numerical data and classify images from the MNIST dataset. We begin by using a three-layer feed-forward neural network to fit a given numerical dataset, computing the least-square error for various training and testing subsets. We then extend our focus to image classification, applying Principal Component Analysis (PCA) to reduce dimensionality, followed by classifying the digits using a neural network. We compare the performance of this network to other machine learning models, including LSTM, SVM, and decision tree classifiers.

### Sec. I. Introduction and Overview
#### Introduction:

Machine learning algorithms have become a cornerstone in the field of data analysis and prediction. Among these algorithms, neural networks are particularly notable for their versatility and ability to model complex, non-linear relationships. In this study, we explore the practical application of neural networks, comparing their performance with other popular machine learning models, such as SVM and decision trees. The purpose of this exercise is to provide a comprehensive understanding of the strengths and weaknesses of these algorithms, and how they can be effectively used for different types of data analysis tasks.

#### Overview:

Our project is divided into two main parts.

In the first part, we focus on a numerical dataset, fitting the data using a three-layer feed-forward neural network. Our approach involves splitting the data into training and test sets in different ways, allowing us to observe how the model's performance varies with different training and testing configurations. The least-square error is computed to evaluate the model's performance on both the training and test sets.

In the second part, we transition to a more complex dataset - the MNIST database of handwritten digits. We start by reducing the dimensionality of the images using PCA, retaining the first 20 modes. We then construct a feed-forward neural network to classify the digits. To evaluate the effectiveness of our neural network, we also train LSTM, SVM, and decision tree classifiers on the same dataset, comparing their performance to our neural network.

Through these exercises, we aim to gain insights into the practical application of neural networks and other machine learning models, and how their performance can be influenced by factors such as the type of data and the method of data preprocessing.

###  Sec. II. Theoretical Background
The model training process is essentially an optimization problem where the aim is to minimize the loss function. Here's a step-by-step breakdown of what happens during training:

* **Forward Pass:** During the forward pass, the model makes predictions based on the current values of its parameters (weights and biases). The input data is passed through each layer of the neural network, with each layer performing specific computations using its current parameters and activation function, and passing the output to the next layer.

* **Loss Calculation:** Once the model has made a prediction, the loss (or error) is calculated using a loss function. In your case, it's the Mean Squared Error (MSE) loss, which calculates the average squared difference between the model's predictions and the actual target values. The output is a single number representing the cost associated with the current state of the model.

* **Backward Pass (Backpropagation):** This is where the model learns. The error calculated in the previous step is propagated back through the network, starting from the final layer. This process involves applying the chain rule to compute the gradient (or derivative) of the loss function with respect to the model parameters. In essence, it determines how much each parameter contributed to the error.

* **Parameter Update:** Once the gradients are computed, they are used to adjust the model parameters in a way that decreases the loss function. The adjustments are made in the opposite direction of the gradients. This is where the learning rate comes into play—it determines the size of the steps taken in the direction of the negative gradient during these updates. The optimizer, like Adam or SGD in our case, is the algorithm that performs these updates.

* **Iterate:** Steps 1 to 4 constitute a single training iteration, or epoch. This process is repeated for a specified number of epochs, with the aim of progressively reducing the loss on our training data.

Over time, this process adjusts the model's parameters so that it can map the input data to the correct output more accurately.


Optimizers play a crucial role in training neural networks by updating the weights and biases during the backpropagation process. Each optimizer uses a different strategy to update the parameters, and some strategies can be more effective than others for a particular problem. For example, some optimizers may converge faster or to a better solution than others. Therefore, selecting the appropriate optimizer for a given problem can have a significant impact on the performance of the model. It's common practice to experiment with different optimizers to find the one that works best for the specific task at hand. We have used Adam and Stochastic Gradient Descent so let us compare to the two:

**Stochastic Gradient Descent (SGD):**

Gradient Descent is a general function optimization algorithm, and Stochastic Gradient Descent is a specific type of it. SGD estimates the error gradient for the current state of the model using examples from the training dataset, then updates the weights of the model using this estimated error.

The word 'stochastic' means a system or a process that is linked with a random probability. Hence, in Stochastic Gradient Descent, a few samples are selected randomly instead of the whole data set for each iteration. In SGD, one update takes place for each training sample, i.e., we do not sum up the gradients for all training examples.

While this can lead to a lot of noise in the training process, it has two primary benefits:

* It can make the training process much faster for large datasets.
* The noise can help the model escape shallow local minima in the loss landscape.

**Adam (Adaptive Moment Estimation) Optimizer:**

Adam is an adaptive learning rate optimization algorithm that's been designed specifically for training deep neural networks. It combines elements from two other extensions of SGD - AdaGrad (Adaptive Gradient Algorithm) and RMSProp (Root Mean Square Propagation).

Adam calculates an exponential moving average of the gradient and the squared gradient, and the parameters beta1 and beta2 control the decay rates of these moving averages.

The initial learning rate is adapted based on how quickly we want parameters to change, which is controlled by the step size, also known as the learning rate.

The benefits of using Adam include:

* Fairly low memory requirements (though higher than SGD and Nesterov Accelerated Gradient).
* Usually works well even with little tuning of hyperparameters.
  
In practice, Adam is currently recommended as the default algorithm to use, and often works slightly better than SGD. However, it is often also beneficial to try out different optimization algorithms for different problems, as there is no one-size-fits-all optimizer.

It is also important to note that nerual networks use data structures called **Tensors**:

In PyTorch, a **Tensor** is a multi-dimensional array that is similar to the ndarray object in NumPy, with the addition being that it can run on a GPU for faster computing. Tensors are fundamental to PyTorch and are used for building neural networks and other machine learning models. They can have different ranks or dimensions, such as a 1D tensor (vector), 2D tensor (matrix), 3D tensor (volume), or higher-dimensional tensors. Each dimension of a tensor is called an axis or a rank, and it has a corresponding size that specifies the number of elements in that axis.


When we start working on the MNIST dataset, we project our data into 20-component PCA space to make our classification easier and have the computation consume less time thanks to a more defined feature space. It is important to understand PCA before moving on:

**Principal Component Analysis (PCA)** is a technique used for dimensionality reduction or feature extraction. It's commonly used in exploratory data analysis and for making predictive models. It is a statistical procedure that orthogonally transforms the 'n' coordinates of a dataset into a new set of 'n' coordinates known as the principal components.

Here's a more detailed explanation:

* **Data Standardization:** PCA starts with a dataset of possibly correlated variables. For successful PCA, it is essential to standardize these initial variables to have mean=0 and variance=1.

* **Covariance Matrix Computation:** PCA computes the covariance matrix of the data to understand how the variables of the input dataset are varying from the mean with respect to each other, or in other words, to see if there is any pattern in the scatter of the data.

* **Eigenvalues and Eigenvectors Calculation:** PCA aims to find the directions (or vectors) that maximize the variance of the data. These directions are called eigenvectors, and the length of an eigenvector is called an eigenvalue. The eigenvector with the highest corresponding eigenvalue is the first principal component.

* **Data Projection:** The last step is to project the original data into these new coordinates (or onto the new basis), giving you the final output of the PCA.

The principal components are a straight line, and the first principal component holds the most variance in the data. Thus, in feature space, this means that the first principal component contains the most dominant features in our data set. Each subsequent principal component is orthogonal to the last and has a lesser variance. In this way, PCA converts the data into a new coordinate system, and the axes of this new system are the principal components.

PCA is often used before applying a machine learning algorithm, to reduce the dimensionality and thus complexity of the dataset, which can help to avoid overfitting, improve model performance, and allow for better visualizations of the data.

### Sec. III. Algorithm Implementation and Development

Initally, I imported all the useful libraries (note that this list grew as I went on with the project):

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
```

Then, I defined a three-layer neural network class:

```
class ThreeLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

The class **ThreeLayerNN** is a subclass of **nn.Module**, which is the base class for all neural network modules in PyTorch. Our class inherits all the properties and methods of nn.Module.

During the initialization (__init__) of an object of this class, three layers are defined:

* **fc1** is a fully connected layer (nn.Linear) that takes input_size inputs and outputs hidden_size outputs.
* **relu** is a Rectified Linear Unit (ReLU) activation function. It is a common activation function in deep learning models that helps introduce non-linearity into the model.
* **fc2** is another fully connected layer that takes hidden_size inputs (from the previous layer) and outputs output_size outputs.

The **forward** function defines the forward propagation of the neural network, i.e., how the data flows through the network from input to output:

* The input x is passed through the first fully connected layer **fc1** and the output is then passed through the **ReLU** activation function.
* This result is then passed through the second fully connected layer **fc2** to produce the final output.

Then, the data was defined using Numpy arrays and converted to PyTorch Tensors:

```
X_np = np.arange(0, 31)
Y_np = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
              40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

X = torch.tensor(X_np, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(Y_np, dtype=torch.float32).view(-1, 1)
```
**X_np** and **Y_np** are 1D arrays, and **X** and **Y** are transformed to 2D tensors with one column each, i.e., column vectors. The **.view()** function in PyTorch is used to reshape the tensor. It's similar to the **reshape()** function in NumPy. The -1 in .view(-1, 1) is a placeholder that tells PyTorch to calculate the correct dimension given the other specified dimensions and the total size of the tensor.

For example, we have a tensor of size (31,), and thus calling .view(-1, 1) on it would reshape it to a size of (31, 1) so that it is now 2D.

Next, I split the data into training and test sets and defined some hyperparameters for my three-layer neural network:

```
# Split the data into training and test sets
X_train_tensorA = X[:20]
Y_train_tensorA = Y[:20]
X_test_tensorA = X[20:]
Y_test_tensorA = Y[20:]

input_size = 1
hidden_size = 32
output_size = 1
learning_rate = 0.01
epochs = 1000

model = ThreeLayerNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

The NN was chosen to have 1-dimensional tensor input and output since the input is a single scalar value and the predicted output should be so as well. The hidden size is the number of nodes in the hidden layer between the input and output layers. The learning rate is set to 0.01 and coresponds to the step size at each iteration while moving toward a minimum of a loss function. A smaller learning rate could make the learning process slower but more precise. An epoch is a single pass through the entire training dataset and after playing around, 1000 seemed to do pretty well. The criterion sets the loss function to be the mean squared error (MSE) and the optimizer is Adam. I chose Adam over SGD since the data here is relatively small and simple, and Adam typically performs better on such datasets.

I then trained the NN and collected the loss:

```
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensorA)
    loss = criterion(outputs, Y_train_tensorA)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    train_outputsA = model(X_train_tensorA)
    train_lossA = criterion(train_outputsA, Y_train_tensorA)
    print(f'Least-square error on training data: {train_lossA.item():.4f}')

    test_outputsA = model(X_test_tensorA)
    test_lossA = criterion(test_outputsA, Y_test_tensorA)
    print(f'Least-square error on test data: {test_lossA.item():.4f}')
```

This code block is the main training loop of the model. Here's how it works:

1. `for epoch in range(epochs):` This begins the training loop. For each epoch (a single pass through the entire training dataset), the model parameters are adjusted to minimize the loss function.

2. `optimizer.zero_grad()` Before the gradients are calculated for this new pass, they need to be explicitly set to zero. This is because PyTorch accumulates gradients, i.e., the gradients calculated for the parameters are added to any previously calculated gradients. For each new pass, we want to start with fresh gradients.

3. `outputs = model(X_train_tensorA)` The forward pass is carried out by passing the training data to the model. The model returns its predictions based on the current state of its parameters.

4. `loss = criterion(outputs, Y_train_tensorA)` The loss function, `criterion`, calculates the loss by comparing the model's predictions, `outputs`, with the actual values, `Y_train_tensorA`.

5. `loss.backward()` The backward pass is initiated. This computes the gradient of the loss with respect to the model parameters. In other words, it calculates how much a small change in each model parameter would affect the loss.

6. `optimizer.step()` The optimizer adjusts the model parameters based on the gradients computed during the `.backward()` call. The learning rate controls the size of these adjustments.

7. `if (epoch+1) % 100 == 0: print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')` This is a logging statement that prints out the loss every 100 epochs. This can help us monitor the training process and see if the loss is decreasing as expected.

Through this repeated process of making predictions, calculating loss, computing gradients and updating parameters, the model learns to make more accurate predictions. This is the essence of training a neural network.

### Sec. IV. Computational Results
### Sec. V. Summary and Conclusions
