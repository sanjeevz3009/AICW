import numpy as np


class MultiLayerNN(object):
    """_summary_"""

    def __init__(self):
        """
        Initialiases the neural network with the weights and biases.
        """
        # self.weight_input_hidden and self.weight_hidden_output are weight
        # matrices connecting the input to the hidden layer and the hidden layer
        # to the output layer
        self.weight_input_hidden = np.random.rand(2, 2)
        # self.bias_hidden and self.bias_output are bias vectors for the hidden and
        # output layers
        self.bias_hidden = np.zeros((1, 2))
        self.weight_hidden_output = np.random.rand(2, 1)
        self.bias_output = np.zeros((1, 1))

    def sigmoid(self, x):
        """
        Sigmoid activation function

        :param x: _description_
        :type x: _type_
        :return: _description_
        :rtype: _type_
        """
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Calculates the sigmoid derivative

        :param x: _description_
        :type x: _type_
        :return: _description_
        :rtype: _type_
        """
        return x * (1.0 - x)

    def feed_forward(self, x):
        """
        feed_forward method performs the forward pass through the network.
        It calculates the inputs to the hidden layer (self.hidden_input) and the
        corresponding outputs (self.hidden_output) using the sigmoid activation
        function.
        Then, it calculates the inputs to the output layer (self.output_input) and the
        final outputs (self.output) using the sigmoid activation function.

        :param x: _description_
        :type x: _type_
        :return: _description_
        :rtype: _type_
        """
        # Forward pass through the network
        self.hidden_input = np.dot(x, self.weight_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = (
            np.dot(self.hidden_output, self.weight_hidden_output) + self.bias_output
        )
        self.output = self.sigmoid(self.output_input)
        return self.output

    def back_propagate(self, input_data, target, learning_rate):
        """
        back_propagate method updates the weights and biases using back propagation.
        It calculates the error at the output layer (output_error) and the corresponding
        deltas.
        Gradients are calculated for both output and hidden layers.
        Weights and biases are updated based on the calculated gradients and the learning rate.

        :param input_data: _description_
        :type input_data: _type_
        :param target: _description_
        :type target: _type_
        :param learning_rate: _description_
        :type learning_rate: _type_
        """
        # Back propagate to update the weights using back propagation
        output_error = target - self.output
        sigmoid_derivative_output = self.sigmoid_derivative(self.output)
        sigmoid_derivative_hidden = self.sigmoid_derivative(self.hidden_output)

        output_delta = output_error * sigmoid_derivative_output
        hidden_delta = (
            np.dot(output_delta, self.weight_hidden_output.T)
            * sigmoid_derivative_hidden
        )

        self.weight_hidden_output += learning_rate * np.outer(
            self.hidden_output, output_delta
        )
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.weight_input_hidden += learning_rate * np.outer(input_data, hidden_delta)
        self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train_network(self, input_data, target_data, epochs, learning_rate):
        """
        train_network method trains the neural network using the provided training data
        (input_data and target_data) for specified number of epochs.
        It iterates through each training example, performs forward and backward passes, and
        updates the parameters.
        The mean squared error (MSE) is calculated for each epoch and printed too.

        :param input_data: _description_
        :type input_data: _type_
        :param target_data: _description_
        :type target_data: _type_
        :param epochs: _description_
        :type epochs: _type_
        :param learning_rate: _description_
        :type learning_rate: _type_
        """
        for epoch in range(epochs):
            total_error = 0
            for input, target_output in zip(input_data, target_data):
                output = self.feed_forward(input)
                self.back_propagate(input, target_output, learning_rate)
                total_error += np.mean((target_output - output) ** 2)
            print(
                f"Epoch {epoch + 1}/{epochs}, Mean Squared Error: {total_error / len(input_data)}"
            )


if __name__ == "__main__":
    # XOR problem training data
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Target outputs for the XOR problem
    target_data = np.array([[0], [1], [1], [0]])

    # OR problem training data
    input_data_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Target outputs for the OR problem
    target_data_or = np.array([0, 1, 1, 1])

    # Create an instance of the MultiLayerNN class, named xor_nn to train the XOR dataset
    xor_nn = MultiLayerNN()
    # The train_network method is called to train the neural network using the XOR training data
    xor_nn.train_network(input_data, target_data, epochs=10000, learning_rate=0.1)

    # Create an instance of the MultiLayerNN class, named or_nn to train the OR dataset
    or_nn = MultiLayerNN()
    # The train_network method is called to train the neural network using the OR training data
    or_nn.train_network(input_data_or, target_data_or, epochs=10000, learning_rate=0.1)

    print("================ Testing the Trained Network for XOR ================")
    for input_data, target_output in zip(input_data, target_data):
        predicted_output = xor_nn.feed_forward(input_data)
        print(
            f"Input: {input_data}, Target Output: {target_output}, Predicted Output: {predicted_output}"
        )
    print("=====================================================================")

    print("\n")

    print("================ Testing the Trained Network for OR ================")
    for input_data, target_output in zip(input_data_or, target_data_or):
        predicted_output = or_nn.feed_forward(input_data)
        print(
            f"Input: {input_data}, Target Output: {target_output}, Predicted Output: {predicted_output}"
        )
    print("=====================================================================")


# The XOR problem is a classic problem where the output is 1 only if exactly one of the inputs is 1. Otherwise, the output is 0.
# The architecture of the neural network is fixed with 2 input neurons, 1 hidden neuron, and 1 output neuron.

# During training, the network learns to map the XOR inputs to the correct outputs.
# The weights and biases are adjusted using the back propagation algorithm to minimise the mean squared error between the predicted and target outputs.

# After training, the network is tested with the same XOR inputs to see how well it generalises to new data.
# The predicted outputs are compared to the target outputs to evaluate the network's performance.

# This code demonstrates the training of a neural network to solve the XOR problem using back propagation.
# The key to its success lies in the presence of a hidden layer, allowing the network to learn and represent non-linear relationships.
