# Importing numpy to be used for numerical operations. i.e Arrays and Matrices
import numpy as np

# Importing numpy typing for type hinting purposes
import numpy.typing as npt


class MultiLayerNN(object):
    """
    Represents a simple neural network.
    """

    def __init__(self):
        """
        Initialiases the neural network with the weights and biases.
        It has 2 input neurons, 1 hidden layer with 2 neurons and 1 output neuron.
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

    def sigmoid(self, x: npt.NDArray):
        """
        Sigmoid activation function.
        Used in the network's forward and backward passes.

        :param x: The sigmoid function takes any real-valued number x and
        squashes it into the range (0, 1). The output is always between 0 and 1
        :type x: npt.NDArray
        :return: The result of this computation, which is a
        value between 0 and 1. The output is the activation of a neuron after
        applying the sigmoid function to its input
        :rtype: NDArray
        """
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: npt.NDArray):
        """
        Calculates the sigmoid derivative.
        Used in the network's forward and backward passes.

        :param x: This is the input to the sigmoid derivative function
        :type x: : npt.NDArray
        :return: The function returns the result of the derivative of the
        sigmoid function applied element-wise to the input array x
        :rtype: npt.NDArray
        """
        return x * (1.0 - x)

    def feed_forward(self, x: npt.NDArray):
        """
        feed_forward method performs the forward pass through the network.
        It calculates the inputs to the hidden layer (self.hidden_input) and the
        corresponding outputs (self.hidden_output) using the sigmoid activation
        function.
        Then, it calculates the inputs to the output layer (self.output_input) and the
        final outputs (self.output) using the sigmoid activation function.

        :param x: This is the input data to the neural network. It represents the features or
        values that you want the neural network to process
        :type x: npt.NDArray
        :return: The function returns the output of the neural network, which represents the
        predicted result after the forward pass
        :rtype: npt.NDArray
        """
        # Forward pass through the network
        # Hidden Layer Input Calculation
        # This calculates the weighted sum of the input (x) to the hidden layer using the weight matrix self.weight_input_hidden
        # and adds the bias term self.bias_hidden
        self.hidden_input = np.dot(x, self.weight_input_hidden) + self.bias_hidden
        # Hidden Layer Output Calculation (Activation)
        # Applies the sigmoid activation function (self.sigmoid) to the hidden layer input, resulting in the output (self.hidden_output)
        self.hidden_output = self.sigmoid(self.hidden_input)
        # Output Layer Input Calculation
        # Calculates the weighted sum of the hidden layer output to obtain the input to the output layer
        # It also includes the bias term
        self.output_input = (
            np.dot(self.hidden_output, self.weight_hidden_output) + self.bias_output
        )
        # Output Layer Output Calculation (Activation)
        # Applies the sigmoid activation function to the output layer input, resulting in the final output
        # of the neural network (self.output)
        self.output = self.sigmoid(self.output_input)
        return self.output

    def back_propagate(
        self, input_data: npt.NDArray, target: npt.NDArray, learning_rate: float
    ):
        """
        back_propagate method updates the weights and biases using back propagation.
        It calculates the error at the output layer (output_error) and the corresponding
        deltas.
        Gradients are calculated for both output and hidden layers.
        Weights and biases are updated based on the calculated gradients and the learning rate.

        :param input_data: The input data that was fed into the network during the forward pass
        :type input_data: npt.NDArray
        :param target: The target or true output corresponding to the input data
        :type target: npt.NDArray
        :param learning_rate: The learning rate hyperparameter, controlling the size of weight updates
        :type learning_rate: float
        """
        # Back propagate to update the weights using back propagation
        # Output Layer Error Calculation
        # Calculates the error at the output layer by taking the difference between the target output and the actual output
        output_error = target - self.output
        # Sigmoid Derivatives
        # Calculates the derivatives of the sigmoid activation function for the output and hidden layers
        # These derivatives are needed for the back propagation calculations
        sigmoid_derivative_output = self.sigmoid_derivative(self.output)
        sigmoid_derivative_hidden = self.sigmoid_derivative(self.hidden_output)

        # Output and Hidden Layer Deltas
        # Computes the deltas for the output and hidden layers
        # These deltas represent the contribution of each layer to the overall error
        output_delta = output_error * sigmoid_derivative_output
        hidden_delta = (
            np.dot(output_delta, self.weight_hidden_output.T)
            * sigmoid_derivative_hidden
        )

        # Weight and Bias Updates
        # Updates the weights and biases based on the calculated deltas and the learning rate
        # This step is crucial for minimising the error in the network by adjusting the parameters
        # in the direction that reduces the error.
        self.weight_hidden_output += learning_rate * np.outer(
            self.hidden_output, output_delta
        )
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.weight_input_hidden += learning_rate * np.outer(input_data, hidden_delta)
        self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train_network(
        self,
        input_data: npt.NDArray,
        target_data: npt.NDArray,
        epochs: int,
        learning_rate: float,
    ):
        """
        train_network method trains the neural network using the provided training data
        (input_data and target_data) for specified number of epochs.
        It iterates through each training example, performs forward and backward passes, and
        updates the parameters.
        The mean squared error (MSE) is calculated for each epoch and printed too.

        :param input_data: The input data used for training
        :type input_data: npt.NDArray
        :param target_data: The target or true output corresponding to the input data
        :type target_data: npt.NDArray
        :param epochs: The number of times to iterate through the entire training dataset
        :type epochs: int
        :param learning_rate: The learning rate hyperparameter, controlling the size of weight updates
        :type learning_rate: float
        """
        # Training Loop
        # Iterates over the specified number of epochs
        for epoch in range(epochs):
            # Initialises the total error for the current epoch to 0
            total_error = 0
            # Iterates through each training example in the dataset
            for input, target_output in zip(input_data, target_data):
                # Performs a forward pass to obtain the output of the neural network for the given input
                output = self.feed_forward(input)
                # Performs the back propagation to update the weights and biases based on the error
                self.back_propagate(input, target_output, learning_rate)
                #  Calculates and accumulates the mean squared error for the current training example
                total_error += np.mean((target_output - output) ** 2)
            # Prints the mean squared error for the current epoch
            # This provides information on how well the neural network is performing on the training data
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


# The XOR problem is a classic problem where the output is 1 only if exactly one of the inputs is 1. Otherwise, the output is 0.
# The architecture of the neural network is fixed with 2 input neurons, 1 hidden neuron, and 1 output neuron.

# During training, the network learns to map the XOR inputs to the correct outputs.
# The weights and biases are adjusted using the back propagation algorithm to minimise the mean squared error between the predicted and target outputs.

# After training, the network is tested with the same XOR inputs to see how well it generalises to new data.
# The predicted outputs are compared to the target outputs to evaluate the network's performance.

# This code demonstrates the training of a neural network to solve the XOR problem using back propagation.
# The key to its success lies in the presence of a hidden layer, allowing the network to learn and represent non-linear relationships.
