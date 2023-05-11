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


In the second part of the project, I compare a feed forward neural network to an LSTM neural network, SVC, and a Decision Tree Classifier:

1. **Long Short-Term Memory (LSTM):**

LSTM is a type of Recurrent Neural Network (RNN) architecture. Traditional RNNs have difficulties in learning long-distance dependencies due to problems known as vanishing and exploding gradients. LSTM networks are designed to combat these problems. They achieve this by introducing gates and a cell state. The gates control the flow of information into and out of the cell state, ensuring that the network has the capability to learn and remember over long sequences. This makes LSTMs particularly useful for sequence prediction problems, including time series prediction, natural language processing, and more.

2. **Support Vector Machine (SVC):**

Support Vector Machine (SVM) is a powerful and flexible class of supervised algorithms for both classification and regression. The fundamental idea behind SVM is to fit the widest possible "street" between the classes; in other words, the goal is to find a decision boundary that maximizes the margin between the closest points (support vectors) of the classes in the training data. SVC stands for Support Vector Classifier which is a type of SVM used for classification tasks. Kernel trick is another important concept in SVM which allows it to solve non-linearly separable problems by transforming the data into higher dimensions.

3. **Decision Tree Classifier:**

A Decision Tree Classifier is a simple yet powerful classification model. The decision tree builds classification or regression models in the form of a tree structure. In a decision tree, each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label. The paths from root to leaf represent classification rules. The primary challenge in the decision tree implementation is to identify which attributes do we need to consider as the root node and each level. This process is done using some statistical approaches like Gini Index, or Gain in Entropy. Decision Trees are easy to understand, and their decisions are interpretable.


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

Next, I split the data into training (first 20) and test (last 10) sets and defined some hyperparameters for my three-layer neural network:

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

For the next part, I split the data into different training (first and last 10) and test (middle 10) sets and then repeat the exact same training process:

```
# Split the data into training and test sets
X_train_tensorB = torch.cat((X[:10], X[-10:]), dim=0)
Y_train_tensorB = torch.cat((Y[:10], Y[-10:]), dim=0)
X_test_tensorB = X[10:20]
Y_test_tensorB = Y[10:20]

# Train the neural network
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensorB)
    loss = criterion(outputs, Y_train_tensorB)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Calculate the least-square error on training and test data
with torch.no_grad():
    train_outputsB = model(X_train_tensorB)
    train_lossB = criterion(train_outputsB, Y_train_tensorB)
    print(f'Least-square error on training data: {train_lossB.item():.4f}')

    test_outputsB = model(X_test_tensorB)
    test_lossB = criterion(test_outputsB, Y_test_tensorB)
    print(f'Least-square error on test data: {test_lossB.item():.4f}')
```

The results of the neural networks were then compared to polynomial fitting results from my non-linear optimization project. I tested against a straight line, a parabola, and a 19th degree polynomial:

```
    # Split data into training and test sets
    X_train_npA = X_np[:20]
    Y_train_npA = Y_np[:20]
    X_test_npA = X_np[20:]
    Y_test_npA = Y_np[20:]

    # Fit a line to the training data
    line_coeffs_trainA = np.polyfit(X_train_npA, Y_train_npA, deg=1)
    line_predictions_trainA = np.polyval(line_coeffs_trainA, X_train_npA)
    line_error_trainA = np.sqrt(np.sum((Y_train_npA - line_predictions_trainA) ** 2)/20)

    # Fit a parabola to the training data
    parabola_coeffs_trainA = np.polyfit(X_train_npA, Y_train_npA, deg=2)
    parabola_predictions_trainA = np.polyval(parabola_coeffs_trainA, X_train_npA)
    parabola_error_trainA = np.sqrt(np.sum((Y_train_npA - parabola_predictions_trainA) ** 2)/20)

    # Fit a 19th degree polynomial to the training data
    poly19_coeffs_trainA = np.polyfit(X_train_npA, Y_train_npA, deg=19)
    poly19_predictions_trainA = np.polyval(poly19_coeffs_trainA, X_train_npA)
    poly19_error_trainA = np.sqrt(np.sum((Y_train_npA - poly19_predictions_trainA) ** 2)/20)

    # Compute errors on test data
    line_predictions_testA = np.polyval(line_coeffs_trainA, X_test_npA)
    line_error_testA = np.sqrt(np.sum((Y_test_npA - line_predictions_testA) ** 2)/10)

    parabola_predictions_testA = np.polyval(parabola_coeffs_trainA, X_test_npA)
    parabola_error_testA = np.sqrt(np.sum((Y_test_npA - parabola_predictions_testA) ** 2)/10)

    poly19_predictions_testA = np.polyval(poly19_coeffs_trainA, X_test_npA)
    poly19_error_testA = np.sqrt(np.sum((Y_test_npA - poly19_predictions_testA) ** 2)/10
```

Since all three of these fits are polynomials, the optimal coefficients were found using Numpy library's polyfit method. An initial guess was not required for this method since polynomials have a known number of solutions and therefore it iteratively minimizes the sum of squares of the residuals between the data and the polynomial fit until it determines the best-fit coefficients.

The same was repeated for the other training (first and last 10) and test (middle 10) set.

After that, I started to use more complicated and higher dimensional data to see how the neural network would compare to famous classification methods such as SVM. 

I began by loading the MNIST train and test data and then contiued to fit the training data into a 20 component PCA space:

```
# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.view(-1))])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_data = mnist_train.data.numpy()
train_data = train_data.reshape(train_data.shape[0], -1)  # Flatten the images

pca = PCA(n_components=20)
pca.fit(train_data)

# The PCA modes are stored in the components_ attribute
pca_modes = pca.components_

# Transform the data into the 20-component PCA space
train_data_pca = pca.transform(train_data)
train_labels = mnist_train.targets.numpy()

test_data = mnist_test.data.numpy()
test_data = test_data.reshape(test_data.shape[0], -1)  # Flatten the images
test_data_pca = pca.transform(test_data)
test_labels = mnist_test.targets.numpy()
```

I then wrote a small helper function that would compute the accuracy of the model:

```
def compute_accuracy(model, x, y, is_torch_model=True):
    with torch.no_grad():
        if is_torch_model:
            x = torch.from_numpy(x).float()
            output = model(x)
            predicted = torch.argmax(output, dim=1).numpy()
        else:
            predicted = model.predict(x)
        accuracy = accuracy_score(y, predicted)
    return accuracy
```
I then created a two-layer feed forward neural network called FFNN:

```
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

1. `self.fc1 = nn.Linear(input_size, hidden_size)`: This is the first layer of the network, often referred to as the input or hidden layer. It applies a linear transformation to the incoming data: `y = xA^T + b`. The input data is transformed from `input_size` dimensions to `hidden_size` dimensions.

2. `self.fc2 = nn.Linear(hidden_size, output_size)`: This is the second layer of the network, often referred to as the output layer. It applies another linear transformation, this time from `hidden_size` dimensions to `output_size` dimensions.

Note that while this network is composed of two layers, it doesn't include any hidden layers in the traditional sense, because there are no layers "hidden" between the input and output layers. 

Also, note that the activation function (ReLU) is not typically counted as a layer. The activation function is applied element-wise and doesn't change the dimensionality of its input.

The model was then trained and tested on the MNIST training and test data in a similar fashion to what was done in the previous part above.

```
for epoch in range(epochs):
    for i, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model_ffnn(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

accuracy_ffnn = compute_accuracy(model_ffnn, test_data_pca, test_labels)
print()
print(f'Feed-forward neural network accuracy: {accuracy_ffnn:.4f}')
```

Finally, I compared the feed forward neural network to an LSTM model, SVC, and Decision Tree Classifier. First I had to create the LSTM neural network and train it as before:

```
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model_lstm = LSTM(input_size, hidden_size, output_size)
optimizer = optim.Adam(model_lstm.parameters(), lr=learning_rate)

# Train the LSTM model
for epoch in range(epochs):
    for i, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model_lstm(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

accuracy_lstm = compute_accuracy(model_lstm, test_data_pca, test_labels)
print()
print(f'LSTM accuracy: {accuracy_lstm:.4f}')
```

After gathering the results for LSTM, I then trained SVC, and a Decision Tree Classifier respectively:

```
model_svm = SVC(kernel='rbf')
model_svm.fit(train_data_pca, train_labels)
accuracy_svm = compute_accuracy(model_svm, test_data_pca, test_labels, is_torch_model=False)
print(f'SVM accuracy: {accuracy_svm:.4f}')

model_dt = DecisionTreeClassifier()
model_dt.fit(train_data_pca, train_labels)
accuracy_dt = compute_accuracy(model_dt, test_data_pca, test_labels, is_torch_model=False)
print(f'Decision tree accuracy: {accuracy_dt:.4f}')
```


### Sec. IV. Computational Results

![image](https://github.com/NajibHaidar/Neural-Networks/assets/116219100/0d3bfba3-5d94-4b88-bd94-5a9dfec4f0d8)
*Figure 1: Three-Layer NN Model Trainng and Test Results on Data A*

In figure 1, we can see that the neural network did not do such a good job at modeling our data A (train: first 20, test: last 10). Our mdodel predicted a straight line although out actual data is similar to a cosine wave. The output of our model strongly depends on its complexity, the optimization process, and the quality of the data it was trained on. Here are a few possibilities of why I believe the model might be predicting a straight line:

1. **Underfitting:** Our model might be too simple to capture the cosine-like patterns in the data. Our model is a linear model or a shallow neural network, it might not have enough flexibility to fit to non-linear data. In this case, we might need to try a more complex model (like a deeper neural network) or engineer more complex features that can capture the cosine-like behavior.

2. **Poor Training:** If the learning rate is too high or too low, or if the model isn't trained for enough epochs, the optimization process might not converge to a good solution. I tryied adjusting the learning rate and training for more epochs but that did not improve the performance.

3. **Initialization:** Neural networks are initialized with random weights. Sometimes, due to bad luck, they might start in a position that leads to poor performance. When I tried reinitializing the model and training again, the results changed, but not for the better or worse; they were mostly random.

4. **Lack of Non-Linearity:** Our model only includes one non-linear activation function, ReLU. Due to lack of more non-linear activation functions (like another ReLU, a sigmoid, or tanh), it might not be able to model non-linear relationships between the input and the output.

All in all, experimenting with hyperparameters may improve results but according to my experience, this was the best I could get and hence I have to condlude that the data is simply too simple and scarce.


![image](https://github.com/NajibHaidar/Neural-Networks/assets/116219100/cc529abe-4eee-4934-adf1-ad12bdd01e88)
*Figure 2: Three-Layer NN Model Trainng and Test Results on Data B*

Figure 2 represents the results of modeling our data B (train: first and last 10, test: middle 10). Similarly to figure 1, the model inferred a line and my hypothesis remains unchanged. It may be noted that the model seemed to do better on this data but I believe this was a coincidence due to the test data having a gap in the middle. This means that a line would do better in making that connection since its as if we have tried to connect two endpoints (area of first 10 points, area of last 10 points).

![image](https://github.com/NajibHaidar/Neural-Networks/assets/116219100/b855d9fd-703f-4ab6-9a2a-fc15bfcae2c9)
*Figure 3: Line Model Trainng and Test Results on Data A*

![image](https://github.com/NajibHaidar/Neural-Networks/assets/116219100/7c59106b-0fd5-4d8e-9436-4c9810f244e2)
*Figure 4: Parabola Model Trainng and Test Results on Data A*

![image](https://github.com/NajibHaidar/Neural-Networks/assets/116219100/b8892ce5-9b16-4821-a264-833d8de77758)
*Figure 5: 19th Degree Polynomial Model Trainng and Test Results on Data A*

![image](https://github.com/NajibHaidar/Neural-Networks/assets/116219100/93695d22-0c7e-428f-9549-7a784406be74)
*Figure 6: Line Model Trainng and Test Results on Data B

![image](https://github.com/NajibHaidar/Neural-Networks/assets/116219100/9ea4322d-a33d-4b43-a29a-6657ed3f6d5d)
*Figure 7: Parabola Model Trainng and Test Results on Data B

![image](https://github.com/NajibHaidar/Neural-Networks/assets/116219100/18ccda10-e6c0-4b7b-b980-9efa20dd7eb2)
*Figure 8: 19th Degree Polynomial Model Trainng and Test Results on Data B

![image](https://github.com/NajibHaidar/Neural-Networks/assets/116219100/cd299f9e-583c-45b3-bb8f-e459bf27a962)
*Figure 9: np.polyfit() LSE Model Comparison on Data A

![image](https://github.com/NajibHaidar/Neural-Networks/assets/116219100/711c0163-161d-4a01-8596-036db7e09870)
*Figure 10: np.polyfit() LSE Model Comparison on Data B

Comparing figures 3-8:

Recall that the given data points oscillate but still steadily increase as X increases.

For the line model, both types of tests yielded very low error in training and testing. However it seems that this model adapted better (although slim) to the version where we removed the middle values during training. This could be interpreted that since we are drawing a line here, the incline is what is important. The total incline can be better depicted by taking points in the beginning and then end of the entire data set and thus this would reduce error.

For the parabola model, the test error after training on the last 10 points was noticeably higher than that on the middle points. This could be interpreted that since the degree of curvature of a parabola will depend on future points, this was better captured when taking points in the beginning and then end of the entire data set and thus this would reduce error.

For the 19th degree polynomial model, the training error in both cases was almost nonexistent. This makes sense since a 19th degree polynomial can pass through all 20 data points with ease due to its degree. However, the test error was astronomical when testing the last 10 points and very large when testing the middle 10 points. This greatly descibes a phenomenon called overfitting. The model has been trained so strongly on the training data (by passing through each point) that when it is given data outside this data set its behaviour is very offset. The test on the middle points was definitely much better (although still not good) than the test on the end points. The interpretation for this is that since the problem here is overfitting, the gap presented by skipping the middle 10 points in training allows this model to be less overfit than its counterpart. The one tested on the first 20 points catches onto the given data points much more strongly and thus results in a much stronger effect of overfitting thus greatly increasing the error when the data changes from what the model was trained on.

The comparision of results of using np.polyfit() vs my three-layer NN can be sumarized as such:

1. **Model Complexity:** np.polyfit() is a linear method that can be used to fit polynomial models. Depending on the degree of the polynomial chosen, this method can model a variety of simple to moderately complex data. The neural network, on the other hand, has the capability to model very complex, high-dimensional, and non-linear relationships. For the cosine-like data, a polynomial of sufficient degree might be able to model the data better than a simple three-layer network.

2. **Training Data Size:** If more data were present, the neural network could potentially perform better. However, this also depends on the complexity of the data. If the underlying data pattern is simple and can be captured by a polynomial, more data might not necessarily improve the neural network's performance significantly.

3. **Determinism:** It's true that np.polyfit() will return the same result each time for the same data and degree, while a neural network may not due to random weight initialization and the stochastic nature of the optimization process. However, this doesn't necessarily mean that the neural network is inferior. Rather, it highlights the importance of using techniques like cross-validation to assess the model's performance and robustness.

4. **Overfitting and Regularization:** It's worth mentioning that while np.polyfit() can fit higher degree polynomials, doing so can lead to overfitting (like in the 19th degree polynomial case), especially with small datasets. The neural network model, despite its poor performance in this case, can leverage regularization techniques like dropout or weight decay, to prevent overfitting.

5. **Error Calculation:** Lastly, it's important to remember that the error calculation is a significant part of model evaluation. Different models may optimize for different types of errors, which might result in different best-fit parameters.

### Sec. V. Summary and Conclusions
