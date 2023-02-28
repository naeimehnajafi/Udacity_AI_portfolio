## Error Term Calculation in Neural Networks

This code is from the Udacity Deep Learning Nanodegree program, specifically from the "Neural Networks" section. It includes a function to calculate the error term for backpropagation in a neural network.

The `error_term_formula` function takes in the input `x`, target output `y`, and actual output `output` of the neural network as arguments, and calculates the error term using the sigmoid activation function. The derivative of the sigmoid activation function is also included in the calculation.

This function can be used as a building block for implementing backpropagation in a neural network, which involves propagating the error backwards through the layers of the network to update the weights and biases.

Note that this code is provided for educational purposes only and may not be suitable for production use without modification.

## Predicting Student Admissions with Neural Networks

The purpose of this code is to predict student admissions to graduate school at UCLA based on three pieces of data: GRE scores, GPA scores, and class rank. It uses a neural network to make these predictions.

The code loads the dataset and formats it using the Pandas and NumPy packages. It then preprocesses the data by normalizing the numerical values and converting the categorical value (class rank) to one-hot encoding.

The neural network is implemented using the NumPy package and includes an input layer, a hidden layer, and an output layer. The sigmoid activation function is used in the hidden layer, while the output layer uses a linear activation function. The error term is calculated using the `error_term_formula` function, which incorporates the derivative of the sigmoid activation function.

Finally, the network is trained using stochastic gradient descent and the mean squared error loss function. The code also includes a function to calculate the accuracy of the predictions.

Overall, the purpose of this code is to demonstrate how to build and train a neural network for predicting student admissions based on numerical and categorical data.
