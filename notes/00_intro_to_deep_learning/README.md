# Introduction to Deep Learning
Deep learning is machine learning that uses lots of data to solve complex problems normally handled by humans. Deep learning is essential to solving "perception" problems, like vision and natural language.

## Topics
 - Perceptrons
 - Softmax
 - One-Hot encoding
 - Maximum Likelihood
 - Logistic regression
 - Gradient descent
 - Non-linear models
 - Feedforward
 - Backpropagation

## Perceptrons
Perceptrons are the building blocks of Neural Networks. If we compare a Neural Network with the brain, perceptrons would be the neurons. They work on a set of inputs and produces an output in the same way a neuron works. What they do is the following: Given a set of inputs and weights (the contribution of each input to the final result) they return an answer to the question we are asking. The simplest question we can ask is if an element belongs to a binary classification or not.
<img src="./images/perceptron.png">

### Major components of a perceptron:
 - **Input**: All the features become the input for a perceptron. We denote the input of a perceptron by [x1, x2, x3, ..,xn], where x   represents the feature value and n represents the total number of features. We also have special kind of input called the bias. In the image, we have described the value of the BIAS as w0.
 - **Weights**: The values that are computed over the time of training the model. Initially, we start the value of weights with some initial value and these values get updated for each training error. We represent the weights for perceptron by [w1,w2,w3,.. wn].
 - **Bias**: A bias neuron allows a classifier to shift the decision boundary left or right. In algebraic terms, the bias neuron allows a classifier to translate its decision boundary. It aims to "move every point a constant distance in a specified direction." Bias helps to train the model faster and with better quality.
 - **Weighted summation**: Weighted summation is the sum of the values that we get after the multiplication of each weight \[wn] associated with the each feature value \[xn]. We represent the weighted summation by ∑wixi for all i -> [1 to n].
 - **Step/activation function**: The role of activation functions is to make neural networks nonlinear. For linear classification, for example, it becomes necessary to make the perceptron as linear as possible.
 - **Output**: The weighted summation is passed to the step/activation function and whatever value we get after computation is our predicted output.
### Perceptron algorithm
 1. Firstly, the features for an example are given as input to the perceptron.
 2. These input features get multiplied by corresponding weights (starting with initial value).
 3. The summation is computed for the value we get after multiplication of each feature with the corresponding weight.
 4. The value of the summation is added to the bias.
 5. The step/activation function is applied to the new value.

### Implementation
```
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(X.shape[0]):
        pred = prediction(X[i],W,b)
        if y[i] > pred:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i] < pred:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
```

## Softmax




## Refrences
 - https://www.coursera.org/learn/intro-to-deep-learning
 - https://github.com/mbadry1/DeepLearning.ai-Summary