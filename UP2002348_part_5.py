import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 data
def load_cifar10(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

# Load the first batch of CIFAR-10 data
cifar_data = load_cifar10('cifar-10-batches-py/data_batch_1')

# Extract images and labels
images = cifar_data[b'data']
labels = np.array(cifar_data[b'labels'])

# Normalize the images
images = images / 255.0

# One-hot encode the labels
num_classes = 10
labels_one_hot = np.eye(num_classes)[labels]

# Neural network architecture
input_size = 32 * 32 * 3
hidden_size = 64
output_size = num_classes

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.randn(input_size, hidden_size)
biases_input_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size)
biases_hidden_output = np.zeros((1, output_size))

# Hyperparameters
learning_rate = 0.01
epochs = 10

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Mean Squared Error loss and its gradient
def mean_squared_error(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred) ** 2)

def mean_squared_error_derivative(y_true, y_pred):
    return y_pred - y_true

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(images, weights_input_hidden) + biases_input_hidden
    hidden_output = sigmoid(hidden_input)
    output = np.dot(hidden_output, weights_hidden_output) + biases_hidden_output

    # Compute loss
    loss = mean_squared_error(labels_one_hot, output)

    # Backward pass (Backpropagation)
    output_error = mean_squared_error_derivative(labels_one_hot, output)
    hidden_error = np.dot(output_error, weights_hidden_output.T) * sigmoid_derivative(hidden_output)

    # Update weights and biases using Gradient Descent
    weights_hidden_output -= learning_rate * np.dot(hidden_output.T, output_error)
    biases_hidden_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
    weights_input_hidden -= learning_rate * np.dot(images.T, hidden_error)
    biases_input_hidden -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

    # Print loss for monitoring training progress
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

# Testing the trained network on a few samples
sample_indices = np.random.choice(images.shape[0], size=5, replace=False)
sample_images = images[sample_indices]
sample_labels = labels[sample_indices]

# Forward pass on the sample images
sample_hidden_input = np.dot(sample_images, weights_input_hidden) + biases_input_hidden
sample_hidden_output = sigmoid(sample_hidden_input)
sample_output = np.dot(sample_hidden_output, weights_hidden_output) + biases_hidden_output
sample_output_softmax = softmax(sample_output)

# Predicted labels
predicted_labels = np.argmax(sample_output_softmax, axis=1)

# Display sample images with true and predicted labels
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axes[i].imshow(sample_images[i].reshape(32, 32, 3))
    axes[i].set_title(f"True: {sample_labels[i]}, Predicted: {predicted_labels[i]}")
    axes[i].axis('off')
plt.show()

fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i in range(8):
    for j in range(8):
        template = weights_input_hidden[:, i * 8 + j].reshape(32, 32, 3)
        axes[i, j].imshow(template)
        axes[i, j].axis('off')
plt.suptitle('Learned Templates in Hidden Layer', fontsize=16)
plt.show()