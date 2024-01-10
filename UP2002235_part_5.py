import pickle
import numpy as np
import urllib.request
import tarfile

# Function to load CIFAR-10 dataset
def load_cifar10():
    # url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    # filename = "cifar-10-python.tar.gz"
    # urllib.request.urlretrieve(url, filename)

    # with tarfile.open(filename, "r:gz") as tar:
    #     tar.extractall()
    
    # Load training data
    with open("cifar-10-batches-py/data_batch_1", 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    
    X_train = data[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    y_train = np.array(data[b'labels'])

    # Load testing data
    with open("cifar-10-batches-py/test_batch", 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    
    X_test = data[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    y_test = np.array(data[b'labels'])
    
    return (X_train, y_train), (X_test, y_test)

# Function to preprocess the data (normalize and flatten)
def preprocess_data(X):
    X = X.astype('float32') / 255.0
    return X.reshape(X.shape[0], -1)

# One-hot encode the labels
def one_hot_encode(y, num_classes):
    encoded = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        encoded[i, y[i]] = 1
    return encoded

# Sigmoid activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Softmax activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-10)) / m

# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        dz2 = self.a2 - y
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = np.dot(dz2, self.weights2.T) * (self.a1 * (1 - self.a1))
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1
        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2

# Training function
def train_neural_network(X_train, y_train, num_epochs, learning_rate, input_size, hidden_size, output_size):
    num_classes = np.max(y_train) + 1
    y_train_encoded = one_hot_encode(y_train, num_classes)

    neural_net = NeuralNetwork(input_size, hidden_size, output_size)

    for epoch in range(num_epochs):
        # Forward pass
        y_pred = neural_net.forward(X_train)

        # Compute loss
        loss = cross_entropy_loss(y_train_encoded, y_pred)

        # Backward pass and update weights
        neural_net.backward(X_train, y_train_encoded, learning_rate)

        # Print loss for every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return neural_net

# Test the neural network on the test set
def test_neural_network(neural_net, X_test, y_test):
    y_pred = neural_net.forward(X_test)
    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy}")

# Main program
if __name__ == "__main__":
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = load_cifar10()

    # Preprocess the data
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)

    # Define neural network parameters
    input_size = X_train.shape[1]
    hidden_size = 256
    output_size = 10
    num_epochs = 100
    learning_rate = 0.01

    # Train the neural network
    neural_net = train_neural_network(X_train, y_train, num_epochs, learning_rate, input_size, hidden_size, output_size)

    # Test the neural network
    test_neural_network(neural_net, X_test, y_test)
