import numpy as np
import neural_network.no_tensor.activation_functions as afuncs
import neural_network.no_tensor.optimizers as optimizers

from typing import Sequence, Type
from neural_network.no_tensor.activation_functions import Array, ActivationFunction, LossFunction
from neural_network.no_tensor.optimizers import Optimizer
from neural_network.data_point import DataPoint



class NeuralNetwork:
    __slots__ = ('layer_count', 'layer_sizes', 'weights', 'biases', 'weight_gradient', 'bias_gradient', 'activation_functions', 'loss_function', 'optimizer')

    EPSILON = 1e-8

    def __init__(self, layer_sizes: Sequence[int], activation_functions: Sequence[ActivationFunction], loss_function: LossFunction, optimizer: Type[Optimizer]) -> None:
        self.layer_count = len(layer_sizes) - 1
        self.layer_sizes = tuple(layer_sizes)

        self.weights = [np.random.uniform(-1 / layer_sizes[i], 1 / layer_sizes[i], (layer_sizes[i + 1], layer_sizes[i])) for i in range(self.layer_count)]
        self.biases = [np.zeros(layer_sizes[i + 1]) for i in range(self.layer_count)]

        self.weight_gradient = [np.zeros(layer.shape) for layer in self.weights]
        self.bias_gradient = [np.zeros(layer.shape) for layer in self.biases]

        self.activation_functions = tuple(activation_functions)
        self.loss_function = loss_function

        self.optimizer = optimizer(self.weights, self.biases)

    def process(self, input: Array) -> Array:
        for i in range(self.layer_count):
            input = self.activation_functions[i](self.biases[i] + self.weights[i].dot(input))

        return input

    def update_gradients(self, input: Array, expected_output: Array) -> None:
        # feed the input through the network to get the values after each layer
        weighted_inputs: list[Array] = []
        node_values = [input]

        for i in range(self.layer_count):
            weighted_inputs.append(self.biases[i] + self.weights[i].dot(node_values[-1]))
            node_values.append(self.activation_functions[i](weighted_inputs[-1]))

        # feed the difference between the output and the expected back through the network, and update the parameters based on their derivatives
        derivative = afuncs.LFUNC_DERIVATIVES[self.loss_function](node_values[-1], expected_output) * afuncs.AFUNC_DERIVATIVES[self.activation_functions[-1]](weighted_inputs[-1])

        self.weight_gradient[-1] += np.outer(derivative, node_values[-2])
        self.bias_gradient[-1] += derivative

        for i in range(self.layer_count - 2, -1, -1):
            derivative = derivative.dot(self.weights[i + 1]) * afuncs.AFUNC_DERIVATIVES[self.activation_functions[i]](weighted_inputs[i])

            self.weight_gradient[i] += np.outer(derivative, node_values[i])
            self.bias_gradient[i] += derivative

    def apply_gradients(self, learning_rate: float) -> None:
        for i in range(self.layer_count):
            weigth_gradient, bias_gradient = self.optimizer(i, self.weight_gradient[i], self.bias_gradient[i])

            self.weights[i] -= weigth_gradient * learning_rate
            self.biases[i] -= bias_gradient * learning_rate

        self.weight_gradient = [np.zeros((self.layer_sizes[i + 1], self.layer_sizes[i])) for i in range(self.layer_count)]
        self.bias_gradient = [np.zeros(self.layer_sizes[i + 1]) for i in range(self.layer_count)]

    def train(self, data_points: list[DataPoint], learning_rate: float, *, noise: float = 0) -> None:
        if noise == 0:
            for data_point in data_points:
                self.update_gradients(data_point.input.data, data_point.expected_output.data)
        else:
            for data_point in data_points:
                self.update_gradients(data_point.noisy_input(noise).data, data_point.expected_output.data)

        self.apply_gradients(learning_rate / len(data_points))

    def get_fitness(self, data_points: Sequence[DataPoint]) -> float:
        total = 0

        for data_point in data_points:
            output = self.process(data_point.input.data)

            cross_entropy = self.loss_function(output, data_point.expected_output.data)

            total += cross_entropy

        return 100 / (total / len(data_points) + 1)



class Classifier(NeuralNetwork):
    '''
    Special network type for classification problems.

    The output layer's activation function should always be `nn.afuncs.softmax`
    '''

    def get_fitness(self, data_points: Sequence[DataPoint]) -> float:
        total = 0

        for data_point in data_points:
            output = self.process(data_point.input.data)

            cross_entropy = self.loss_function(output, data_point.expected_output.data)

            total += cross_entropy

        average_cross_entropy = total / len(data_points)

        return 100 * (1 - average_cross_entropy)

    def get_accuracy(self, data_points: Sequence[DataPoint]) -> float:
        correct = 0

        for data_point in data_points:
            output = self.process(data_point.input.data)

            correct += data_point.expected_output.data[np.argmax(output)]

        return 100 * correct / len(data_points)

