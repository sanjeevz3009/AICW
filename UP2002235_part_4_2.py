import numpy as np

class MultiLayerNN(object):
    """_summary_"""

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

    def sigmoid(self, x):
        """
        Sigmoid activation function
        """
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Calculates the sigmoid derivative
        """
        return x * (1.0 - x)

    def feed_forward(self, x):
        """
        feed_forward method performs the forward pass through the network.
        """
        # Forward pass through the network
        self.hidden1_input = np.dot(x, self.weight_input_hidden1) + self.bias_hidden1
        self.hidden1_output = self.sigmoid(self.hidden1_input)
        
        self.hidden2_input = np.dot(self.hidden1_output, self.weight_hidden1_hidden2) + self.bias_hidden2
        self.hidden2_output = self.sigmoid(self.hidden2_input)
        
        self.output_input = np.dot(self.hidden2_output, self.weight_hidden2_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)
        
        return self.output

    def back_propagate(self, input_data, target, learning_rate):
        """
        back_propagate method updates the weights and biases using back propagation.
        """
        # Back propagate to update the weights using back propagation
        output_error = target - self.output
        sigmoid_derivative_output = self.sigmoid_derivative(self.output)
        
        hidden2_error = np.dot(output_error, self.weight_hidden2_output.T)
        sigmoid_derivative_hidden2 = self.sigmoid_derivative(self.hidden2_output)
        
        hidden1_error = np.dot(hidden2_error, self.weight_hidden1_hidden2.T)
        sigmoid_derivative_hidden1 = self.sigmoid_derivative(self.hidden1_output)

        output_delta = output_error * sigmoid_derivative_output
        hidden2_delta = hidden2_error * sigmoid_derivative_hidden2
        hidden1_delta = hidden1_error * sigmoid_derivative_hidden1

        self.weight_hidden2_output += learning_rate * np.outer(self.hidden2_output, output_delta)
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.weight_hidden1_hidden2 += learning_rate * np.outer(self.hidden1_output, hidden2_delta)
        self.bias_hidden2 += learning_rate * np.sum(hidden2_delta, axis=0, keepdims=True)
        
        self.weight_input_hidden1 += learning_rate * np.outer(input_data, hidden1_delta)
        self.bias_hidden1 += learning_rate * np.sum(hidden1_delta, axis=0, keepdims=True)

    def train_network(self, input_data, target_data, epochs, learning_rate):
        """
        train_network method trains the neural network using the provided training data
        for specified number of epochs.
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
