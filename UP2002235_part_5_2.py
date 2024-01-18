import numpy as np
import pickle
import matplotlib.pyplot as plt


# Load CIFAR-10 data
def unpickle(file):
    with open("cifar-10-batches-py/data_batch_1", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = []
labels = []

for i in range(1, 6):
    batch_file = f'cifar-10-batches-py/data_batch_{i}'
    batch_data = unpickle(batch_file)
    data.append(batch_data[b'data'])
    labels.extend(batch_data[b'labels'])

X = np.vstack(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
y = np.array(labels)

# Split the data into training and testing sets
split_ratio = 0.8
split_idx = int(len(X) * split_ratio)

X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define a simple neural network with templates
class NeuralNetwork:
    def __init__(self):
        self.templates = np.random.rand(10, 32, 32, 3)

    def predict(self, X):
        # Calculate similarity between input images and templates
        similarity_scores = np.zeros((X.shape[0], 10))
        for i in range(10):
            similarity_scores[:, i] = np.sum(np.abs(X - self.templates[i]), axis=(1, 2, 3))

        # Predict the class with the highest similarity
        predictions = np.argmin(similarity_scores, axis=1)
        return predictions

# Initialize the neural network
nn = NeuralNetwork()

# Demonstration of 'templates' in the neural network
print("Template Images:")
for i in range(10):
    template_image = (nn.templates[i] * 255).astype(np.uint8)
    plt.subplot(2, 5, i + 1)
    plt.imshow(template_image)
    plt.title(f'Template {i + 1}')
    plt.axis('off')

plt.show()

# Demonstration of the correct use of test data and displaying predicted images
print("\nTesting the Neural Network:")
predictions = nn.predict(X_test)

# Display some random training and predicted images
num_display = 5
selected_indices = np.random.choice(len(X_test), num_display, replace=False)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(selected_indices):
    plt.subplot(2, num_display, i + 1)
    plt.imshow(X_test[idx])
    plt.title(f'True: {y_test[idx]}')
    plt.axis('off')

    plt.subplot(2, num_display, num_display + i + 1)
    plt.imshow(nn.templates[predictions[idx]])
    plt.title(f'Predicted: {predictions[idx]}')
    plt.axis('off')

plt.show()
