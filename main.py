import numpy as np
import pandas
from NeuralNetwork import Network, relu, relu_derivative, sigmoid, sigmoid_derivative, cross_entropy, cross_entropy_derivative

Train = pandas.read_csv(".gitignore/mnist_train.csv")
Test = pandas.read_csv(".gitignore/mnist_test.csv")
X= Train.iloc[3000:6000, 1:].values / 255.0
Y= np.eye(10)[Train.iloc[3000:6000, 0].values]

X_test = Test.iloc[1000:2000, 1:].values / 255.0
Y_test = np.eye(10)[Test.iloc[1000:2000, 0].values]




network = Network(loss_func=cross_entropy, loss_func_derivative=cross_entropy_derivative)
network.add_layer(784, 128, relu, relu_derivative)
network.add_layer(128, 64, relu, relu_derivative)
network.add_layer(64, 10, sigmoid, sigmoid_derivative)

epochs = 10
learning_rate = 0.01
batch_size = 10

print("Starting training on MNIST dataset...")
network.train(X, Y, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

predictions = network.predict(X_test)

accuracy = np.mean(predictions == np.argmax(Y_test, axis=1))
print(f"Test Accuracy on MNIST: {accuracy * 100:.2f}%")

network.save_model("mnist_trained_model.pkl")
