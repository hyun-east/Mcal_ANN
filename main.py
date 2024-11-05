import numpy as np
import pandas
from NeuralNetwork import *
#import matplotlib.pyplot as plt

Train = pandas.read_csv("mnist_train.csv")
Test = pandas.read_csv("mnist_test.csv")
X= Train.iloc[:, 1::4].values / 255.0
Y= np.eye(10)[Train.iloc[:, 0].values]

X_test = Test.iloc[:, 1::4].values / 255.0
Y_test = np.eye(10)[Test.iloc[:500, 0].values]


network = Network(loss_func=cross_entropy, loss_func_derivative=cross_entropy_derivative)
network.add_layer(196, 128, leakyrelu, leakyrelu_derivative)
network.add_layer(128, 64, leakyrelu, leakyrelu_derivative)
network.add_layer(64, 10, sigmoid, sigmoid_derivative)

epochs = 100
learning_rate = 0.001
batch_size = 64

print("Starting training on MNIST dataset...")
network.train(X, Y, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

predictions = network.predict(X_test)

#accuracy = np.mean(predictions == np.argmax(Y_test, axis=1))
#print(f"Test Accuracy on MNIST: {accuracy * 100:.2f}%")

network.save_model("mnist_trained_model.pkl")

