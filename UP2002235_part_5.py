import pickle

import numpy as np


# Load CIFAR-10 dataset
def load_cifar10():
    """
    Reads the CIFAR-10 dataset from a file named 'data_batch_1' using the pickle module.
    It returns the raw pixel data and corresponding labels.
    """
    with open("cifar-10-batches-py/data_batch_1", "rb") as file:
        batch = pickle.load(file, encoding="latin1")
    return batch["data"], batch["labels"]


# Normalise the data
def normalise_data(data):
    """
    Scales the pixel values in the dataset to the range [0, 1] by dividing them by 255.
    """
    return data / 255.0


# One-hot encode labels
def one_hot_encode(labels):
    """
    Converts the class labels into one-hot encoded vector.
    """
    num_classes = len(np.unique(labels))
    encoded = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded[i, label] = 1
    return encoded


# Sigmoid activation function
def sigmoid(x):
    """
    Implements the sigmoid activation function, which squashes the input values between 0 and 1.
    """
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid function
def sigmoid_derivative(x):
    """
    Computes the derivative of the sigmoid function.
    """
    return x * (1 - x)


# Neural network class
class NeuralNetwork:
    """
    Represent a simple feedforward neural network.
    It has methods for training (train) and making predictions (predict).
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialises the weights and biases with random values.
        """
        # Initialise weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def train(self, X, y, epochs, learning_rate):
        """
        Performs training using back propagation.
        It takes input data (X), corresponding labels (y), the number of epochs, and the learning rate as parameters.
        """
        for epoch in range(epochs):
            print("Epoch", epoch)
            # Forward pass
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = sigmoid(hidden_layer_input)

            output_layer_input = (
                np.dot(hidden_layer_output, self.weights_hidden_output)
                + self.bias_output
            )
            predicted_output = sigmoid(output_layer_input)

            # Calculate loss
            loss = y - predicted_output

            # Backpropagation
            output_error = loss * sigmoid_derivative(predicted_output)
            hidden_layer_error = output_error.dot(
                self.weights_hidden_output.T
            ) * sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            self.weights_hidden_output += (
                hidden_layer_output.T.dot(output_error) * learning_rate
            )
            self.bias_output += (
                np.sum(output_error, axis=0, keepdims=True) * learning_rate
            )

            self.weights_input_hidden += X.T.dot(hidden_layer_error) * learning_rate
            self.bias_hidden += (
                np.sum(hidden_layer_error, axis=0, keepdims=True) * learning_rate
            )

    def predict(self, X):
        """
        Performs a forward pass to generate predictions for input data.
        """
        # Forward pass for prediction
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = (
            np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        )
        predicted_output = sigmoid(output_layer_input)

        return predicted_output


def load_cifar10_meta():
    """
    It returns a list of class names associated with the CIFAR-10 dataset.
    This list is typically used to map numeric labels to human-readable class names for better interpretation of the results.
    """
    with open("cifar-10-batches-py/batches.meta", "rb") as file:
        meta = pickle.load(file, encoding="latin1")
    return meta["label_names"]


# Load CIFAR-10 dataset and class names
data, labels = load_cifar10()
class_names = load_cifar10_meta()

# Normalise and one-hot encode the data
data = normalise_data(data)
labels_one_hot = one_hot_encode(labels)

# Map class names to numerical labels
class_name_to_label = {class_name: i for i, class_name in enumerate(class_names)}

# Choose the class to train on (e.g., "airplane")
train_class_name = "airplane"
train_class_label = class_name_to_label[train_class_name]

# Filter data and labels for the chosen class
train_class_indices = np.where(np.array(labels) == train_class_label)[0]
train_class_data = data[train_class_indices]
train_class_labels_one_hot = labels_one_hot[train_class_indices]

# Select the number of images to train on
num_images_to_train = 100  # Specify the number of images to train on

# Initialize neural network
input_size = data.shape[1]
hidden_size = 64
output_size = len(class_names)
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network on the specified number of images
for i in range(min(num_images_to_train, len(train_class_indices))):
    image_to_train = train_class_data[i].reshape(1, -1)
    label_to_train = train_class_labels_one_hot[i].reshape(1, -1)

    # Train for a small number of epochs for demonstration purposes
    epochs = 100
    learning_rate = 0.01
    nn.train(image_to_train, label_to_train, epochs, learning_rate)

# Specify the class of images to test
test_class_name = "airplane"

# Filter data and labels for the chosen test class
test_class_label = class_name_to_label[test_class_name]
test_class_indices = np.where(np.array(labels) == test_class_label)[0]
test_class_data = data[test_class_indices]
test_class_labels_one_hot = labels_one_hot[test_class_indices]

# Specify the number of images to test
num_images_to_test = 10  # Specify the number of images to test

# Test the trained neural network on the specified number of images from the test class
for i in range(min(num_images_to_test, len(test_class_indices))):
    test_image = test_class_data[i].reshape(1, -1)
    predicted_output = nn.predict(test_image)
    predicted_label = np.argmax(predicted_output)

    # Calculate accuracy for each test image
    accuracy = (predicted_label == test_class_label) * 100.0

    # Print the actual and predicted labels along with class names and accuracy
    actual_label = np.argmax(test_class_labels_one_hot[i])
    actual_class_name = class_names[actual_label]

    predicted_class_name = class_names[predicted_label]

    print(f"Testing on class '{test_class_name}', Image {i + 1}:")
    print("Actual Class Name:", actual_class_name)
    print("Predicted Class Name:", predicted_class_name)
    print("Accuracy:", accuracy, "%")
    print()
