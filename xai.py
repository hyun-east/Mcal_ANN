import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def leakyrelu(x):
    return np.maximum(-0.1 * x, x)


def leakyrelu_derivative(x):
    return np.where(x > 0, 1, -0.1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def cross_entropy(Y_real, Y_calc):
    epsilon = 1e-15
    Y_calc = np.clip(Y_calc, epsilon, 1 - epsilon)
    return -np.mean(Y_real * np.log(Y_calc) + (1 - Y_real) * np.log(1 - Y_calc))


def cross_entropy_derivative(Y_real, Y_calc):
    epsilon = 1e-15
    Y_calc = np.clip(Y_calc, epsilon, 1 - epsilon)
    return -(Y_real / Y_calc) + ((1 - Y_real) / (1 - Y_calc))


# 레이어 클래스
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

    # 가중치와 편향 기울기 계산
    def compute_weight_bias_gradients(self, input_data):
        # Ensure input_data is in 2D form and matches input size of the layer
        input_data = input_data.reshape(1, -1)  # Reshape to match layer's input size

        # Calculate gradient with respect to weights and bias
        grad_W = np.dot(input_data.T, np.ones_like(self.output))  # Match shape to self.weight
        grad_b = np.sum(self.output, axis=0)  # Bias gradient summed over batch

        return grad_W, grad_b



# 네트워크 클래스
class Network:
    def __init__(self, loss_func=cross_entropy, loss_func_derivative=cross_entropy_derivative):
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
        if Y_pred.shape[1] > 1:
            return np.mean(np.argmax(Y, axis=1) == np.argmax(Y_pred, axis=1))
        else:
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

    def compute_data_contribution(self, input_data, baseline=None, steps=50):
        if baseline is None:
            baseline = np.zeros_like(input_data)
        alphas = np.linspace(0, 1, steps)
        data_contributions = np.zeros_like(input_data)

        for alpha in alphas:
            interpolated_input = baseline + alpha * (input_data - baseline)
            gradients = self.compute_gradients(interpolated_input)
            data_contributions += gradients / steps

        return (input_data - baseline) * data_contributions

    def compute_structure_contribution(self, input_data, baseline=None, steps=50):
        if baseline is None:
            baseline = np.zeros_like(input_data)
        alphas = np.linspace(0, 1, steps)
        structure_contributions = []

        for layer in self.layers:
            layer_contribution = 0

            input_data_layer = input_data
            baseline_layer = baseline[:input_data_layer.size].reshape(1, -1)

            for alpha in alphas:
                interpolated_input = baseline_layer + alpha * (input_data_layer.reshape(1, -1) - baseline_layer)

                layer_output = layer.forward(interpolated_input)

                grad_W = np.dot(interpolated_input.T, np.ones_like(layer_output))
                grad_b = np.sum(layer_output, axis=0)

                layer_contribution += (np.sum(layer.weight * grad_W) + np.sum(layer.bias * grad_b)) / steps

            structure_contributions.append(layer_contribution)

            input_data = layer_output

        return np.array(structure_contributions)

    def compute_joint_contribution(self, data_contributions, structure_contributions, sigma=0.5):
        joint_contribution = 0
        for i in range(len(data_contributions)):
            for j in range(len(structure_contributions)):
                kernel = np.exp(-((i - j) ** 2) / (2 * sigma ** 2))
                joint_contribution += data_contributions[i] * structure_contributions[j] * kernel
        return joint_contribution

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

    def compute_gradients(self, input_data):
        # Forward pass to calculate the output
        input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)
        output = self.forward(input_data)

        gradients = np.zeros_like(input_data)
        output_gradient = np.ones_like(output)

        for layer, (_, activation_derivative) in reversed(list(zip(self.layers, self.activations))):
            if activation_derivative is not None:
                output_gradient *= activation_derivative(layer.output)
            output_gradient = layer.backward(output_gradient, learning_rate=0)

        gradients = output_gradient

        return gradients.flatten()


# 시각화 함수들
def visualize_original_image(input_data, ax, input_shape=(28, 28), prediction=None):
    input_image = input_data.reshape(input_shape)
    ax.imshow(input_image, cmap='gray')
    title = "Original Image" if prediction is None else f"Pred: {prediction}"
    ax.set_title(title)
    ax.axis('off')


def visualize_positive_contribution(data_contribution, ax, input_shape=(28, 28)):
    positive_contribution = np.maximum(data_contribution, 0).reshape(input_shape)
    ax.imshow(positive_contribution, cmap='hot', interpolation='nearest')
    ax.set_title("Positive Contribution")
    ax.axis('off')


def visualize_negative_contribution(data_contribution, ax, input_shape=(28, 28)):
    negative_contribution = np.minimum(data_contribution, 0).reshape(input_shape)
    ax.imshow(-negative_contribution, cmap='cool', interpolation='nearest')  # 음수는 절대값으로 표시
    ax.set_title("Negative Contribution")
    ax.axis('off')


def visualize_difference(data_contribution, structure_contribution, ax, input_shape=(28, 28)):
    # 구조적 기여도를 입력 차원과 일치하도록 확장
    structure_contribution_expanded = np.full(data_contribution.shape, structure_contribution[0])
    difference = (data_contribution - structure_contribution_expanded).reshape(input_shape)

    # 차이 시각화
    ax.imshow(difference, cmap='bwr', interpolation='nearest', vmin=-np.max(np.abs(difference)),
              vmax=np.max(np.abs(difference)))
    ax.set_title("Difference (Data - Structure)")
    ax.axis('off')


# 실행 부분
if __name__ == "__main__":
    # MNIST 데이터 로드 및 전처리
    Train = pd.read_csv("mnist_train.csv")
    Test = pd.read_csv("mnist_test.csv")

    X = Train.iloc[:, 1:].values / 255.0
    Y = Train.iloc[:, 0].values  # 숫자 라벨

    # 모델 생성 및 학습
    network = Network(loss_func=cross_entropy, loss_func_derivative=cross_entropy_derivative)
    network.add_layer(784, 128, leakyrelu, leakyrelu_derivative)
    network.add_layer(128, 64, leakyrelu, leakyrelu_derivative)
    network.add_layer(64, 10, sigmoid, sigmoid_derivative)

    epochs = 10
    learning_rate = 0.001
    batch_size = 64
    print("Starting training on MNIST dataset...")
    network.train(X, np.eye(10)[Y], epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

    # 숫자별로 여러 샘플 비교
    unique_digits = range(10)  # 0~9 숫자
    samples_per_digit = 10  # 각 숫자별로 시각화할 샘플 수

    for digit in unique_digits:
        # 각 숫자에 대한 샘플 추출
        digit_indices = np.where(Y == digit)[0]

        fig, axes = plt.subplots(samples_per_digit, 4, figsize=(16, 4 * samples_per_digit))
        fig.suptitle(f"Digit: {digit}", fontsize=16)

        for j in range(samples_per_digit):
            sample_index = digit_indices[j]  # 여러 샘플 선택
            input_data = X[sample_index]

            # 자료 중심 기여도 계산
            data_contribution = network.compute_data_contribution(input_data)

            # 구조 중심 기여도 계산
            structure_contribution = network.compute_structure_contribution(input_data)

            # 예측값 계산
            prediction = np.argmax(network.forward(input_data.reshape(1, -1)))

            # 각 샘플에 대해 4개의 시각화
            visualize_original_image(input_data, axes[j, 0], prediction=prediction)
            visualize_positive_contribution(data_contribution, axes[j, 1])
            visualize_negative_contribution(data_contribution, axes[j, 2])
            visualize_difference(data_contribution, structure_contribution, axes[j, 3])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
