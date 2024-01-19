# Importing typing module for type hinting purposes
from typing import List

# Importing numpy to be used for numerical operations. i.e Arrays and Matrices
import numpy as np

# Importing numpy typing for type hinting purposes
import numpy.typing as npt


# Perceptron class definition
class Perceptron:
    """
    The Perceptron class encapsulates the essential components of a perceptron, including
    its weights, bias, prediction mechanism, and training procedure.
    """

    def __init__(self, input_size: int):
        """
        Initialises the perceptron with random weights and a bias term.

        :param input_size: How many inputs to handle. In this case, 2 because
            the logical operations (XOR and OR) involve two binary inputs
        :type input_size: int
        """
        # Initialise weights with small random values
        self.weights = np.random.rand(input_size)
        # Set a bias term
        self.bias = np.random.rand()
        print("Generate random weights and bias term:")
        print("Weights: ", self.weights)
        print("Bias: ", self.bias)
        print("\n")

    def predict(self, inputs: List) -> int:
        """
        Predict method that takes input values, calculates the weighted sum (dot product of inputs and weights)
        plus the bias, and then applies a step function as the activation function.
        The step function outputs 1 if the weighted sum is greater than or equal to 0, and 0 otherwise.

        :param inputs: The features that the perceptron will use to make a prediction. e.g. [0, 1]
        :type inputs: List
        :return: The predicted output, which is either 1 or 0
        :rtype: int
        """
        # Calculate the weighted sum of inputs and add bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        print("Weighted sum: ", weighted_sum)
        print("\n")
        # Apply step function (threshold activation function)
        return 1 if weighted_sum >= 0 else 0

    def train(
        self,
        training_inputs: npt.NDArray,
        labels: npt.NDArray,
        problem_name: str,
        learning_rate: float = 0.1,
        epochs: int = 100,
    ):
        """
        Train method for training the perceptron. It iterates through the training data for a specified number of epoch,
        makes predictions, and updates the weights and bias based on the error.

        :param training_inputs: The input features of the training data
        :type training_inputs: npt.NDArray
        :param labels: Actual output labels corresponding to each set of the input features in training_inputs
        :type labels: npt.NDArray
        :param learning_rate: Hyperparameter that determines the step size during the weight and bias updates
            , defaults to 0.1
        :type learning_rate: float, optional
        :param epochs: This is the number of times the entire training dataset is passed through
            the perceptron for training, defaults to 100
        :type epochs: int, optional
        """
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            for inputs, label in zip(training_inputs, labels):
                print("Problem name: ", problem_name)
                print("Inputs, labels: ", inputs, label)
                # Predict the output
                prediction = self.predict(inputs)
                print("Prediction: ", prediction)
                # Update weights and bias based on the error
                self.weights += learning_rate * (label - prediction) * inputs
                self.bias += learning_rate * (label - prediction)
                print("Weights: ", self.weights)
                print("Bias: ", self.bias)


# The input arrays below represents all possible combination of input values, and the
# corresponding label arrays provide the expected output

# OR problem training data
or_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_labels = np.array([0, 1, 1, 1])  # OR output

# XOR problem training data
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_labels = np.array([0, 1, 1, 0])  # XOR output

# Create and train a perceptron for the OR problem
# Creates instances of the Perceptron class for the OR and XOR problems, and then trains them using
# the respective training data
or_perceptron = Perceptron(input_size=2)
or_perceptron.train(or_inputs, or_labels, problem_name="OR")

# Create and train a perceptron for the XOR problem
xor_perceptron = Perceptron(input_size=2)
xor_perceptron.train(xor_inputs, xor_labels, problem_name="XOR")

# Test the perceptron for the OR problem
print("\n")
print(
    "As you can see from the below predicted output, the perceptron has learned to solve the OR problem, but not the XOR problem."
)
print("OR Problem:")
for inputs in or_inputs:
    prediction = or_perceptron.predict(inputs)
    print(f"Inputs: {inputs}, Predicted Output: {prediction}")

# Test the perceptron for the XOR problem
print("\n")
print(
    "As you can see from the below predicted output, the perceptron has learned to solve the OR problem, but not the XOR problem."
)
print("The XOR predicted output below is not correct.")
print("XOR Problem:")
for inputs in xor_inputs:
    prediction = xor_perceptron.predict(inputs)
    print(f"Inputs: {inputs}, Predicted Output: {prediction}")
