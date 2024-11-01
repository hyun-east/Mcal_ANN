import numpy as np
import pickle


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def leakyrelu(x):
    return np.maximum(-0.1 * x, x)


def leaky_relu_derivative(x):
    return np.where(x > 0, 1, -0.1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def MSE(Y_real, Y_calc):
    return np.mean((Y_real - Y_calc) ** 2)


def MSE_derivative(Y_real, Y_calc):
    return 2 * (Y_calc - Y_real) / Y_real.size


def cross_entropy(Y_real, Y_calc):
    epsilon = 1e-15
    Y_calc = np.clip(Y_calc, epsilon, 1 - epsilon)
    return -np.mean(Y_real * np.log(Y_calc) + (1 - Y_real) * np.log(1 - Y_calc))


def cross_entropy_derivative(Y_real, Y_calc):
    epsilon = 1e-15
    Y_calc = np.clip(Y_calc, epsilon, 1 - epsilon)
    return -(Y_real / Y_calc) + ((1 - Y_real) / (1 - Y_calc))


class Network:
    def __init__(self, loss_func=MSE, loss_func_derivative=MSE_derivative):
        self.layers = []
        self.activations = []
        self.loss_func = loss_func
        self.loss_func_derivative = loss_func_derivative
        self.loss_history = []
        self.accuracy_history = []

    def add_layer(self, input_size, output_size, activ_func, active_func_diff):
        self.layers.append(Layer(input_size, output_size))
        self.activations.append((activ_func, active_func_diff))

    def forward(self, input_array):
        ret = input_array
        for layer, (active, _) in zip(self.layers, self.activations):
            output = active(layer.forward(ret))
            ret = output
        return ret

    def check_loss(self, X, Y):
        Y_calc = self.forward(X)
        return self.loss_func(Y, Y_calc)

    def compute_accuracy(self, Y, Y_pred):
        if Y_pred.shape[1] > 1:  # 다중 클래스 분류
            return np.mean(np.argmax(Y, axis=1) == np.argmax(Y_pred, axis=1))
        else:  # 이진 분류
            return np.mean((Y == (Y_pred >= 0.5).astype(int)))

    def backward(self, Y, Y_calc, learning_rate):
        output_gradient = self.loss_func_derivative(Y, Y_calc)
        for layer, (_, activation_derivative) in reversed(list(zip(self.layers, self.activations))):
            if activation_derivative is not None:
                output_gradient = activation_derivative(layer.output) * output_gradient
            output_gradient = layer.backward(output_gradient, learning_rate)

    def train(self, X, Y, epochs, learning_rate, batch_size=32):
        num_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X, Y = X[indices], Y[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_X, batch_Y = X[start_idx:end_idx], Y[start_idx:end_idx]

                Y_calc = self.forward(batch_X)
                self.backward(batch_Y, Y_calc, learning_rate)

            epoch_loss = self.check_loss(X, Y)
            epoch_accuracy = self.compute_accuracy(Y, self.forward(X))

            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_accuracy)

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy * 100:.2f}%')

    def predict(self, X):
        output = self.forward(X)
        if output.shape[1] > 1:
            return np.argmax(output, axis=1)
        return (output >= 0.5).astype(int)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model


class Layer:
    def __init__(self, input_size, output_size):
        self.weight = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weight) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        dW = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weight.T)
        self.weight -= learning_rate * dW
        self.bias -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        return input_gradient