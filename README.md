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

Our study is divided into two main parts.

In the first part, we focus on a numerical dataset, fitting the data using a three-layer feed-forward neural network. Our approach involves splitting the data into training and test sets in different ways, allowing us to observe how the model's performance varies with different training and testing configurations. The least-square error is computed to evaluate the model's performance on both the training and test sets.

In the second part, we transition to a more complex dataset - the MNIST database of handwritten digits. We start by reducing the dimensionality of the images using PCA, retaining the first 20 modes. We then construct a feed-forward neural network to classify the digits. To evaluate the effectiveness of our neural network, we also train LSTM, SVM, and decision tree classifiers on the same dataset, comparing their performance to our neural network.

Through these exercises, we aim to gain insights into the practical application of neural networks and other machine learning models, and how their performance can be influenced by factors such as the type of data and the method of data preprocessing.

###  Sec. II. Theoretical Background
The model training process is essentially an optimization problem where the aim is to minimize the loss function. Here's a step-by-step breakdown of what happens during training:

**Forward Pass:** During the forward pass, the model makes predictions based on the current values of its parameters (weights and biases). The input data is passed through each layer of the neural network, with each layer performing specific computations using its current parameters and activation function, and passing the output to the next layer.

**Loss Calculation:** Once the model has made a prediction, the loss (or error) is calculated using a loss function. In your case, it's the Mean Squared Error (MSE) loss, which calculates the average squared difference between the model's predictions and the actual target values. The output is a single number representing the cost associated with the current state of the model.

**Backward Pass (Backpropagation):** This is where the model learns. The error calculated in the previous step is propagated back through the network, starting from the final layer. This process involves applying the chain rule to compute the gradient (or derivative) of the loss function with respect to the model parameters. In essence, it determines how much each parameter contributed to the error.

**Parameter Update:** Once the gradients are computed, they are used to adjust the model parameters in a way that decreases the loss function. The adjustments are made in the opposite direction of the gradients. This is where the learning rate comes into play—it determines the size of the steps taken in the direction of the negative gradient during these updates. The optimizer, like Adam or SGD in our case, is the algorithm that performs these updates.

**Iterate:** Steps 1 to 4 constitute a single training iteration, or epoch. This process is repeated for a specified number of epochs, with the aim of progressively reducing the loss on our training data.

Over time, this process adjusts the model's parameters so that it can map the input data to the correct output more accurately.

Optimizers play a crucial role in training neural networks by updating the weights and biases during the backpropagation process. Each optimizer uses a different strategy to update the parameters, and some strategies can be more effective than others for a particular problem. For example, some optimizers may converge faster or to a better solution than others. Therefore, selecting the appropriate optimizer for a given problem can have a significant impact on the performance of the model. It's common practice to experiment with different optimizers to find the one that works best for the specific task at hand. We have used Adam and Stochastic Gradient Descent so let us compare to the two:

**Stochastic Gradient Descent (SGD):**

Gradient Descent is a general function optimization algorithm, and Stochastic Gradient Descent is a specific type of it. SGD estimates the error gradient for the current state of the model using examples from the training dataset, then updates the weights of the model using this estimated error.

The word 'stochastic' means a system or a process that is linked with a random probability. Hence, in Stochastic Gradient Descent, a few samples are selected randomly instead of the whole data set for each iteration. In SGD, one update takes place for each training sample, i.e., we do not sum up the gradients for all training examples.

While this can lead to a lot of noise in the training process, it has two primary benefits:

**-** It can make the training process much faster for large datasets.
**-** The noise can help the model escape shallow local minima in the loss landscape.

**Adam (Adaptive Moment Estimation) Optimizer:**

Adam is an adaptive learning rate optimization algorithm that's been designed specifically for training deep neural networks. It combines elements from two other extensions of SGD - AdaGrad (Adaptive Gradient Algorithm) and RMSProp (Root Mean Square Propagation).

Adam calculates an exponential moving average of the gradient and the squared gradient, and the parameters beta1 and beta2 control the decay rates of these moving averages.

The initial learning rate is adapted based on how quickly we want parameters to change, which is controlled by the step size, also known as the learning rate.

The benefits of using Adam include:

**-** Fairly low memory requirements (though higher than SGD and Nesterov Accelerated Gradient).
**-** Usually works well even with little tuning of hyperparameters.
  
In practice, Adam is currently recommended as the default algorithm to use, and often works slightly better than SGD. However, it is often also beneficial to try out different optimization algorithms for different problems, as there is no one-size-fits-all optimizer.

When we start working on the MNIST dataset, we project our data into 20-component PCA space to make our classification easier and have the computation consume less time thanks to a more defined feature space. It is important to understand PCA before moving on:


**Principal Component Analysis (PCA)** is a technique used for dimensionality reduction or feature extraction. It's commonly used in exploratory data analysis and for making predictive models. It is a statistical procedure that orthogonally transforms the 'n' coordinates of a dataset into a new set of 'n' coordinates known as the principal components.

Here's a more detailed explanation:

**-Data Standardization:** PCA starts with a dataset of possibly correlated variables. For successful PCA, it is essential to standardize these initial variables to have mean=0 and variance=1.

**-Covariance Matrix Computation:** PCA computes the covariance matrix of the data to understand how the variables of the input dataset are varying from the mean with respect to each other, or in other words, to see if there is any pattern in the scatter of the data.

**-Eigenvalues and Eigenvectors Calculation:** PCA aims to find the directions (or vectors) that maximize the variance of the data. These directions are called eigenvectors, and the length of an eigenvector is called an eigenvalue. The eigenvector with the highest corresponding eigenvalue is the first principal component.

**-Data Projection:** The last step is to project the original data into these new coordinates (or onto the new basis), giving you the final output of the PCA.

The principal components are a straight line, and the first principal component holds the most variance in the data. Thus, in feature space, this means that the first principal component contains the most dominant features in our data set. Each subsequent principal component is orthogonal to the last and has a lesser variance. In this way, PCA converts the data into a new coordinate system, and the axes of this new system are the principal components.

PCA is often used before applying a machine learning algorithm, to reduce the dimensionality and thus complexity of the dataset, which can help to avoid overfitting, improve model performance, and allow for better visualizations of the data.
