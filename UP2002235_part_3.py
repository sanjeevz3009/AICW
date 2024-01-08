import numpy as np

class SingleLayerNN(object):
    def __init__(self):
        # Initialise weights and biases
        self.weight_input_hidden = np.random.rand(2, 2)
        self.bias_hidden = np.zeros((1, 2))
        self.weight_hidden_output = np.random.rand(2, 1)
        self.bias_output = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)
    
    def feed_forward(self, x):
        # Forward pass through the network
        self.hidden_input = np.dot(x, self.weight_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weight_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)
        return self.output

    def back_propagate(self, input_data, target, learning_rate):
        # Back propagate to update the weights using back propagation
        output_error = target - self.output
        sigmoid_derivative_output = self.sigmoid_derivative(self.output)
        sigmoid_derivative_hidden = self.sigmoid_derivative(self.hidden_output)

        output_delta = output_error * sigmoid_derivative_output
        hidden_delta = np.dot(output_delta, self.weight_hidden_output.T) * sigmoid_derivative_hidden

        self.weight_hidden_output += learning_rate * np.outer(self.hidden_output, output_delta)
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.weight_input_hidden += learning_rate * np.outer(input_data, hidden_delta)
        self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    
    def train_network(self, input_data, target_data, epochs, learning_rate):
        for epoch in range(epochs):
            total_error = 0
            for input, target_output in zip(input_data, target_data):
                output = self.feed_forward(input)
                self.back_propagate(input, target_output, learning_rate)
                total_error += np.mean((target_output - output) ** 2)
            print(f"Epoch {epoch + 1}/{epochs}, Mean Squared Error: {total_error / len(input_data)}")
                
if __name__ == "__main__":
    # XOR problem training data
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_data = np.array([[0], [1], [1], [0]])

    # OR problem training data
    input_data_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_data_or = np.array([0, 1, 1, 1])  # OR output

    xor_nn = SingleLayerNN()
    xor_nn.train_network(input_data, target_data, epochs=10000, learning_rate=0.1)

    or_nn = SingleLayerNN()
    or_nn.train_network(input_data_or, target_data_or, epochs=10000, learning_rate=0.1)



    print("================ Testing the Trained Network for XOR ================")
    for input_data, target_output in zip(input_data, target_data):
        predicted_output = xor_nn.feed_forward(input_data)
        print(f"Input: {input_data}, Target Output: {target_output}, Predicted Output: {predicted_output}")
    print("=====================================================================")

    print("\n")

    print("================ Testing the Trained Network for OR ================")
    for input_data, target_output in zip(input_data_or, target_data_or):
        predicted_output = xor_nn.feed_forward(input_data)
        print(f"Input: {input_data}, Target Output: {target_output}, Predicted Output: {predicted_output}")
    print("=====================================================================")
