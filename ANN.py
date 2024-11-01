import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def MSE(Y_real, Y_calc):
    pass

def cross_entrophy(Y_real, Y_calc):
    pass

class Network:
    def __init__(self):
        self.layers = []
        self.activations = []

    def add_layer(self, input_size, output_size, activ_func, active_func_diff):
        self.layers.append(Layer(input_size, output_size))
        self.activations.append((activ_func, active_func_diff))

    def forward(self, input_array):
        ret = input_array
        for layer, (active, _) in zip(self.layers, self.activations):
            output = active(layer.forward(ret))
            ret = output

        return ret

    def check_loss(self,X, Y, loss_func):
        Y_calc = self.forward(X)
        loss = loss_func(Y, Y_calc)
        return loss

    def backward(self):
        pass

class Layer:
    def __init__(self, input_size, output_size):
        self.weight = np.random.randn(input_size, output_size) * 0.1 + 0.1
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        output = np.dot(input, self.weight) + self.bias
        return output

    def backward(self, output_gradient, learning_rate):
        dW = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weight.T)
        self.weight -= learning_rate*dW
        self.bias -= learning_rate*np.sum(output_gradient)
        return input_gradient

Nn = Network()

x = np.array([1, 2, 3, 4, 5])

Nn.add_layer(5, 2, relu, relu_derivative)
Nn.add_layer(2, 2, relu, relu_derivative)
Nn.add_layer(2, 2, relu, relu_derivative)
Nn.add_layer(2, 5, relu, relu_derivative)

print(Nn.forward(x))