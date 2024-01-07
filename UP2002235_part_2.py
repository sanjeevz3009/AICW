# Importing numpy to be used for numerical operations. i.e Arrays and Matrices
import numpy as np

# Perceptron class definition
class Perceptron:
    """
    The Perceptron class encapsulates the essential components of a perceptron, including
    its weights, bias, prediction mechanism, and training procedure.
    """
    def __init__(self, input_size):
        """_summary_

        :param input_size: _description_
        :type input_size: _type_
        """
        # Initialise weights with small random values
        self.weights = np.random.rand(input_size)
        # Set a bias term
        self.bias = np.random.rand()
        print("Weights: ", self.weights)
        print("Bias: ", self.bias)
        print("\n")

    def predict(self, inputs):
        """
        Predict method that takes input values, calculates the weighted sum (dot product of inputs and weights)
        plus the bias, and then applies a step function as the activation function.
        The step function outputs 1 if the weighted sum is greater than or equal to 0, and 0 otherwise.

        :param inputs: _description_
        :type inputs: _type_
        :return: _description_
        :rtype: _type_
        """
        # Calculate the weighted sum of inputs and add bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        print("Weighted sum: ", weighted_sum)
        print("\n")
        # Apply step function (threshold activation function)
        return 1 if weighted_sum >= 0 else 0

    def train(self, training_inputs, labels, learning_rate=0.1, epochs=10):
        """
        Train method for training the perceptron. It iterates through the training data for a specified number of epoch,
        makes predictions, and updates the weights and bias based on the error.

        :param training_inputs: _description_
        :type training_inputs: _type_
        :param labels: _description_
        :type labels: _type_
        :param learning_rate: _description_, defaults to 0.1
        :type learning_rate: float, optional
        :param epochs: _description_, defaults to 100
        :type epochs: int, optional
        """
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            for inputs, label in zip(training_inputs, labels):
                print("Inputs, labels: ", inputs, label)
                # Predict the output
                prediction = self.predict(inputs)
                print("Prediction: ", prediction)
                # Update weights and bias based on the error
                self.weights += learning_rate * (label - prediction) * inputs
                self.bias += learning_rate * (label - prediction)
                print("Weights: ", self.weights)
                print("Bias: ", self.bias)

# The input arrays below represents all possible combination of input values, and the
# corresponding label arrays provide the expected output

# OR problem training data
or_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_labels = np.array([0, 1, 1, 1])  # OR output

# XOR problem training data
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_labels = np.array([0, 1, 1, 0])  # XOR output

# Create and train a perceptron for the OR problem
# Creates instances of the Perceptron class for the OR and XOR problems, and then trains them using
# the respective training data
print("OR")
print("\n")
or_perceptron = Perceptron(input_size=2)
or_perceptron.train(or_inputs, or_labels)

print("XOR")
print("\n")
# Create and train a perceptron for the XOR problem
xor_perceptron = Perceptron(input_size=2)
xor_perceptron.train(xor_inputs, xor_labels)

# Test the perceptron for the OR problem
print("OR Problem:")
for inputs in or_inputs:
    prediction = or_perceptron.predict(inputs)
    print(f"Inputs: {inputs}, Predicted Output: {prediction}")

# Test the perceptron for the XOR problem
print("\nXOR Problem:")
for inputs in xor_inputs:
    prediction = xor_perceptron.predict(inputs)
    print(f"Inputs: {inputs}, Predicted Output: {prediction}")
