import numpy as np
import neural_network.tensor as tensor
import neural_network.activation_functions as afuncs
import neural_network.optimizers as optimizers

from neural_network import Network, Layer
from neural_network.tensor import Tensor
from neural_network.data_point import DataPoint
from neural_network.mnist import MNIST



class DigitRecognizer(Network):
    def __init__(self, *layers: Layer, name: str = '') -> None:
        super().__init__(*layers, name=name)

        self.l1 = layers[0]
        self.l2 = layers[1]

        self.optimizer = optimizers.ADAM(*self.parameters)

    def forward(self, input: Tensor) -> Tensor:
        z1 = afuncs.ReLU(self.l1.forward(input))
        z2 = self.l2.forward(z1)

        return afuncs.softmax(z2)

    def train(self, data_points: list[DataPoint], learning_rate: float) -> None:
        for data_point in data_points:
            out = self.forward(data_point.noisy_input(5))

            loss = afuncs.cross_entropy(out, data_point.expected_output)

            loss.backpropagate()

        self.optimizer.apply_gradients(learning_rate / len(data_points))

    def get_accuracy(self, data_points: list[DataPoint]) -> float:
        correct = 0

        for data_point in data_points:
            out = self.forward(data_point.input)

            correct += data_point.expected_output[np.argmax(out.data)] # type: ignore

        return correct / len(data_points) # type: ignore

    @classmethod
    def new(cls, hidden_layer_dim: int = 100, *, name: str = '') -> 'DigitRecognizer':
        layers = (
            Layer(28*28, hidden_layer_dim, initialization='xavier'),
            Layer(hidden_layer_dim, 10, initialization='xavier')
        )

        return DigitRecognizer(*layers, name=name)



if __name__ == '__main__':
    network = DigitRecognizer.new(100, name='test1.nn')

    mnist = MNIST()

    learning_rate = 1 / 20

    training_batches = 1_000
    training_batch_size = 100

    test_interval = 25
    test_batch_size = 2_000

    for i in range(training_batches):
        if (i % test_interval) == 0:
            accuracy = network.get_accuracy(mnist.get_random_test_images(test_batch_size))

            print(f'Accuracy after {i:^3} batches: {accuracy * 100:.2f}%')

        network.train(mnist.get_random_training_images(training_batch_size), learning_rate)

