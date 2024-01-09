# Importing numpy to be used for numerical operations. i.e Arrays and Matrices
import numpy as np
# Matplotlib is a plotting library used to visualise the training process by plotting
# the Mean Squared Error (MSE)
import matplotlib.pyplot as plt

# Importing typing module for type hinting purposes
from typing import List
# Importing numpy typing for type hinting purposes
import numpy.typing as npt

class SingleLayerNN(object):
    """
    SingleLayerNN class represents a simple neural network with a single layer.
    """
    def __init__(self):
        """
        Initialises the weights and biases randomly
        """
        # self.weight_input_output is a NumPy array of shape (2, 1)
        # 2 corresponds to the number of input features (assuming each data point has two features)
        # 1 corresponds to the number of neurons in the output layer
        # During the training process, these weights are updated using backpropagation to minimize the
        # Mean Squared Error (MSE) between the predicted output and the actual target.
        self.weight_input_output = np.random.rand(2, 1)
        self.bias_output = np.zeros((1, 1))

    def sigmoid(self, x):
        """
        Implements the sigmoid activation function.

        :param x: _description_
        :type x: _type_
        :return: _description_
        :rtype: _type_
        """
        # Sigmoid activation function
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Implements the derivative of the sigmoid activation function.

        :param x: _description_
        :type x: _type_
        :return: _description_
        :rtype: _type_
        """
        # Sigmoid derivative
        return x * (1.0 - x)

    def feed_forward(self, x):
        """
        Performs the forward pass through the network.
        Calculates the output by applying the sigmoid activation function to the
        weighted sum of inputs.

        :param x: Input data
        :type x: _type_
        :return: self.output is computed in the feed_forward method by applying the sigmoid activation function to self.output_input
        :rtype: _type_
        """
        # Forward pass through the network

        # self.output_input is a variable representing the weighted sum of inputs to
        # the output layer before the activation function is applied

        # This is often referred to as the input to the activation function
        # self.output_input is computed in the feed_forward method using the dot product of
        # the input data (x) and the weights connecting the input to the output layer (self.weight_input_output)
        # Additionally, the bias term (self.bias_output) is added
        self.output_input = np.dot(x, self.weight_input_output) + self.bias_output

        # After computing self.output_input, the sigmoid activation function is applied
        # in the feed_forward method to produce the actual output of the neural network
        self.output = self.sigmoid(self.output_input)

        # self.output is computed in the feed_forward method by applying the sigmoid activation function to self.output_input
        # During the training process, this output is compared to the actual target values to calculate the error, and then back propagation
        # is used to adjust the weights and biases in order to minimize this error. The sigmoid activation function introduces non-linearity
        # to the model, allowing the network to learn complex patterns in the data.
        return self.output

    def back_propagate(self, input_data, target, learning_rate):
        """
        Updates weights and biases using back propagation.
        Calculates the error and adjusts the weights and biases accordingly.

        :param input_data: Input data
        :type input_data: _type_
        :param target: Target is the actual target output that the neural network is trying to approximate during training
        :type target: _type_
        :param learning_rate: Learning rate
        :type learning_rate: _type_
        """
        # Based on the back propagation equations
        # dE/DWi =(y - y[i+1]) S'(x[i+1])xi
        # S' (x[i+1])=S(x[i+1])(1-s(x[i+1)))
        # s(x[i+1]=x[i+1]            
        # x[i+1]=yiWi

        # Back propagation to update weights and biases

        # Target is the actual target output that the neural network is trying to approximate during training
        # self.output is the output of the neural network, which is the result of the forward pass

        # The output_error is a measure of how far off the neural network's prediction is from the actual target
        # This error is then used in the back propagation process to adjust the weights and biases of the neural network in
        # the direction that reduces this error
        output_error = target - self.output

        # sigmoid_derivative_output is a variable representing the derivative of the sigmoid activation function with respect to
        # the output of the neural network. Specifically, it is used in the back propagation process to calculate the gradient
        # of the error with respect to the output
        sigmoid_derivative_output = self.sigmoid_derivative(self.output)

        # output_delta is a variable representing the error gradient with respect to the output of the neural network during the
        # back propagation process
        # It is a key component in updating the weights and biases of the network to minimise the error
        output_delta = output_error * sigmoid_derivative_output

        # self.weight_input_output is a variable representing the weights connecting the input layer to the output layer of the
        # neural network
        # These weights are crucial parameters that the network learns during the training process

        # During back propagation, the weights are updated based on the calculated output_delta
        # The outer product of the input data and output_delta is multiplied by the learning rate and added to the existing weights
        # This update rule aims to adjust the weights in a direction that reduces the error

        # The self.weight_input_output matrix is a crucial part of the neural network's parameters that are learned during training
        # These weights determine how much each input feature contributes to the output and are adjusted to capture patterns in the data
        self.weight_input_output += learning_rate * np.outer(input_data, output_delta)

        # This line of code updates the bias term based on the calculated output_delta. The sum of the output_delta along the 
        # specified axis (axis=0) is multiplied by the learning rate and added to the existing bias

        # output_delta represents the gradient of the error with respect to the output of the neural network
        # np.sum(output_delta, axis=0, keepdims=True) calculates the sum of these gradients along axis=0,
        # effectively accumulating the contribution of each training example to the overall change in the bias
        # The sum is then scaled by the learning rate (learning_rate) before being added to the existing bias

        # This update ensures that the bias is adjusted in a direction that reduces the error, contributing to the
        # overall learning of the neural network during the training process. The bias term allows the neural network to
        # introduce an offset or shift in the output, providing flexibility in capturing patterns in the data
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

    # Trains the neural network using input data, target_data, the number of epochs, and
    # learning rate
    def train_network(self, input_data, target_data, epochs, learning_rate):
        mse_values = []
        # Train the neural network
        for epoch in range(epochs):
            total_error = 0
            for input, target_output in zip(input_data, target_data):
                output = self.feed_forward(input)
                self.back_propagate(input, target_output, learning_rate)
                total_error += np.mean((target_output - output) ** 2)
            mse_values.append(total_error / len(input_data))
            print(f"Epoch {epoch + 1}/{epochs}, Mean Squared Error: {total_error / len(input_data)}")

        # Plot the Mean Squared Error
        # The Mean Squared Error is plotted to visualise the training progress.
        # plt.plot(range(1, epochs + 1), mse_values)
        # plt.xlabel('Epoch')
        # plt.ylabel('Mean Squared Error')
        # plt.title('Training Progress')
        # plt.show()


if __name__ == "__main__":
    # XOR problem training data
    # Defines XOR problem training data (input_data) and corresponding
    # outputs (target_data)
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Target outputs for the XOR problem
    target_data = np.array([[0], [1], [1], [0]])

    # Create an instance of the SingleLayerNN class
    single_layer_nn = SingleLayerNN()

    # Trains the neural network on the XOR problem and prints the input, target_output and
    # predicted output
    single_layer_nn.train_network(input_data, target_data, epochs=10000, learning_rate=0.1)

    # Test the trained network
    print("================ Testing the Trained Network ================")
    for input_data, target_output in zip(input_data, target_data):
        predicted_output = single_layer_nn.feed_forward(input_data)
        print(f"Input: {input_data}, Target Output: {target_output}, Predicted Output: {predicted_output}")
    print("================================================================")
