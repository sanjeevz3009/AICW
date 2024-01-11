# Importing numpy to be used for numerical operations. i.e Arrays and Matrices
import numpy as np
# Importing numpy typing for type hinting purposes
import numpy.typing as npt


class MultiLayerNN(object):
    """
    Represents a multo-layer neural network with 2 hidden layers.
    It is initialised with random weights and biases for the connections
    between layers.
    """

    def __init__(self):
        """
        Initialiases the neural network with the weights and biases.
        """
        # self.weight_input_hidden1, self.weight_hidden1_hidden2, and self.weight_hidden2_output
        # are weight matrices connecting the input to the first hidden layer, the first hidden layer
        # to the second hidden layer, and the second hidden layer to the output layer
        self.weight_input_hidden1 = np.random.rand(2, 3)
        self.weight_hidden1_hidden2 = np.random.rand(3, 2)
        self.weight_hidden2_output = np.random.rand(2, 1)

        # self.bias_hidden1, self.bias_hidden2, and self.bias_output are bias vectors
        # for the first hidden layer, the second hidden layer, and the output layer
        self.bias_hidden1 = np.zeros((1, 3))
        self.bias_hidden2 = np.zeros((1, 2))
        self.bias_output = np.zeros((1, 1))

    def relu(self, x: npt.NDArray) -> npt.NDArray:
        """
        Rectified Linear Unit (ReLU) activation function.
        This ReLU activation is commonly used in the hidden layers of neural networks to introduce non-linearity,
        allowing the network to learn and approximate more complex relationships in the data during the training process.

        :param x: Represents the input value or array to which the Rectified Linear Unit (ReLU) activation function is applied
        :type x: npt.NDArray
        :return: The ReLU activation function outputs the input value if it is positive, and zero otherwise
        :rtype: npt.NDArray
        """
        # np.maximum(0, x): This NumPy function element-wise compares each element of the
        # input x with 0 and returns the maximum value between 0 and the input element
        return np.maximum(0, x)

    def relu_derivative(self, x: npt.NDArray) -> npt.NDArray:
        """
        Calculates the derivative of the ReLU activation function.

        :param x: Represents the input value or array for which you want to calculate the derivative of the ReLU activation
        :type x: npt.NDArray
        :return: Returns If an element in x is greater than 0, the corresponding element in the output is set
            to 1; otherwise, it is set to 0
        :rtype: npt.NDArray
        """
        # The function uses NumPy's np.where function to create an array of the same shape as the input x. The elements of this array
        # are determined by the condition x > 0. If an element in x is greater than 0, the corresponding element in the output is set
        # to 1; otherwise, it is set to 0.
        return np.where(x > 0, 1, 0)

    def feed_forward(self, x: npt.NDArray) -> npt.NDArray:
        """
        feed_forward method performs the forward pass through the network.
        During this pass, the input data is processed layer by layer, and the output of each layer becomes the input for the next layer.
        The Rectified Linear Unit (ReLU) activation function is applied to the outputs of the hidden layers.

        :param x: Represents the input data that you want to process through the neural network
        :type x: npt.NDArray
        :return: Returns the output of the neural network after processing the input x through the forward pass
            This output represents the model's prediction or activation values at the output layer
        :rtype: npt.NDArray
        """
        # Forward pass through the network with ReLU activation for hidden layers

        # First Hidden Layer
        # self.hidden1_input: Computes the weighted sum of the input x by the weights self.weight_input_hidden1 and adds the bias self.bias_hidden1
        self.hidden1_input = np.dot(x, self.weight_input_hidden1) + self.bias_hidden1
        # self.hidden1_output: Applies the ReLU activation function to self.hidden1_input
        self.hidden1_output = self.relu(self.hidden1_input)

        # Second Hidden Layer
        # self.hidden2_input: Computes the weighted sum of the first hidden layer's output by the weights self.weight_hidden1_hidden2
        # and adds the bias self.bias_hidden2
        self.hidden2_input = (
            np.dot(self.hidden1_output, self.weight_hidden1_hidden2) + self.bias_hidden2
        )
        # self.hidden2_output: Applies the ReLU activation function to self.hidden2_input
        self.hidden2_output = self.relu(self.hidden2_input)

        # Output Layer
        # self.output_input: Computes the weighted sum of the second hidden layer's output by the weights self.weight_hidden2_output
        # and adds the bias self.bias_output
        self.output_input = (
            np.dot(self.hidden2_output, self.weight_hidden2_output) + self.bias_output
        )
        # self.output: Applies the ReLU activation function to self.output_input
        self.output = self.relu(self.output_input)

        return self.output

    def back_propagate(
        self, input_data: npt.NDArray, target: npt.NDArray, learning_rate: float
    ):
        """
        back_propagate method updates the weights and biases using back propagation.

        :param input_data: The input data for which the network made a prediction during the forward pass
        :type input_data: npt.NDArray
        :param target: The target output corresponding to the provided input data
        :type target: npt.NDArray
        :param learning_rate: The learning rate used to control the step size during weight and bias updates
        :type learning_rate: float
        """
        # Based on the back propagation equations
        # dE/DWi =(y - y[i+1]) S'(x[i+1])xi
        # S' (x[i+1])=S(x[i+1])(1-s(x[i+1)))
        # s(x[i+1]=x[i+1]
        # x[i+1]=yiWi

        # Back propagate to update the weights using back propagation

        # Error Calculation
        output_error = target - self.output

        # Back propagation through Layers
        # relu_derivative_output, relu_derivative_hidden2, relu_derivative_hidden1: Compute the derivatives of the ReLU activation functions
        # for the output and hidden layers
        # hidden2_error, hidden1_error: Compute the errors at the hidden layers by propagating the output error backward through the weights
        relu_derivative_output = self.relu_derivative(self.output)

        # self.weight_hidden2_output.T, self.weight_hidden1_hidden2.T: Transpose weight matrices for use in error calculations
        hidden2_error = np.dot(output_error, self.weight_hidden2_output.T)
        relu_derivative_hidden2 = self.relu_derivative(self.hidden2_output)

        hidden1_error = np.dot(hidden2_error, self.weight_hidden1_hidden2.T)
        relu_derivative_hidden1 = self.relu_derivative(self.hidden1_output)

        # output_delta, hidden2_delta, hidden1_delta: Compute the deltas, which represent the adjustments needed for the weights
        # Update weights and biases using the deltas and the learning rate
        output_delta = output_error * relu_derivative_output
        hidden2_delta = hidden2_error * relu_derivative_hidden2
        hidden1_delta = hidden1_error * relu_derivative_hidden1

        self.weight_hidden2_output += learning_rate * np.outer(
            self.hidden2_output, output_delta
        )
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.weight_hidden1_hidden2 += learning_rate * np.outer(
            self.hidden1_output, hidden2_delta
        )
        self.bias_hidden2 += learning_rate * np.sum(
            hidden2_delta, axis=0, keepdims=True
        )

        self.weight_input_hidden1 += learning_rate * np.outer(input_data, hidden1_delta)
        self.bias_hidden1 += learning_rate * np.sum(
            hidden1_delta, axis=0, keepdims=True
        )

    def train_network(
        self,
        input_data: npt.NDArray,
        target_data: npt.NDArray,
        epochs: int,
        learning_rate: float,
    ):
        """
        train_network method trains the neural network using the provided training data
        for specified number of epochs.

        :param input_data: The input data used for training the neural network
        :type input_data: npt.NDArray
        :param target_data: The target data corresponding to the input data, representing the desired outputs
        :type target_data: npt.NDArray
        :param epochs: The number of training epochs, i.e., the number of times the entire training dataset is passed
            forward and backward through the neural network
        :type epochs: int
        :param learning_rate: he learning rate, a hyperparameter that controls the step size during weight and bias updates
        :type learning_rate: float
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
    training_data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Target outputs for the XOR problem
    training_target_data_xor = np.array([[0], [1], [1], [0]])

    # OR problem training data
    training_data_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Target outputs for the OR problem
    training_target_data_or = np.array([0, 1, 1, 1])

    # Create an instance of the MultiLayerNN class, named xor_nn to train the XOR dataset
    xor_nn = MultiLayerNN()
    # The train_network method is called to train the neural network using the XOR training data
    xor_nn.train_network(
        training_data_xor, training_target_data_xor, epochs=10000, learning_rate=0.1
    )

    # Create an instance of the MultiLayerNN class, named or_nn to train the OR dataset
    or_nn = MultiLayerNN()
    # The train_network method is called to train the neural network using the OR training data
    or_nn.train_network(
        training_data_or, training_target_data_or, epochs=10000, learning_rate=0.1
    )

    print("================ Testing the Trained Network for XOR ================")
    for input_data, target_output in zip(training_data_xor, training_target_data_xor):
        predicted_output = xor_nn.feed_forward(input_data)
        print(
            f"Input: {input_data}, Target Output: {target_output}, Predicted Output: {predicted_output}"
        )
    print("=====================================================================")

    print("\n")

    print("================ Testing the Trained Network for OR ================")
    for input_data, target_output in zip(training_data_or, training_target_data_or):
        predicted_output = or_nn.feed_forward(input_data)
        print(
            f"Input: {input_data}, Target Output: {target_output}, Predicted Output: {predicted_output}"
        )
    print("=====================================================================")
