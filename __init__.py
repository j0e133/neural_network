import numpy as np
import neural_network.tensor as tensor

from typing import Literal
from neural_network.tensor import Tensor, Array



class Layer:
    __slots__ = ('weights', 'biases')

    def __init__(self, input_dim: int, output_dim: int, requires_grad: bool = True, initialization: Literal['he', 'xavier'] = 'xavier') -> None:
        match initialization:
            case 'xavier':
                x = (6 / (input_dim + output_dim)) ** 0.5
                self.weights = Tensor(np.random.uniform(-x, x, (output_dim, input_dim)), requires_grad)
            case 'he':
                std = (2 / input_dim) ** 0.5
                self.weights = Tensor(np.random.normal(0, std, (output_dim, input_dim)), requires_grad)

        self.biases = Tensor(np.zeros(output_dim), requires_grad)

    @property
    def parameters(self) -> tuple[Tensor, Tensor]:
        return self.weights, self.biases

    def forward(self, input: Tensor) -> Tensor:
        return self.weights.dot(input) + self.biases

    def copy(self, requires_grad: bool | None = None) -> 'Layer':
        output = Layer(1, 1)

        output.weights = self.weights.copy(requires_grad)
        output.biases = self.biases.copy(requires_grad)

        return output

    def get_save_data(self) -> bytes:
        input_dim = self.weights.shape[1].to_bytes(2)
        output_dim = self.weights.shape[0].to_bytes(2)
        requires_grad = int(self.weights.requires_grad).to_bytes(1)

        weights = self.weights.data.tobytes()
        biases = self.biases.data.tobytes()

        size = (5 + len(weights) + len(biases)).to_bytes(4)

        return size + input_dim + output_dim + requires_grad + weights + biases

    @classmethod
    def from_bytes(cls, _bytes: bytes) -> 'Layer':
        input_dim = int.from_bytes(_bytes[:2])
        output_dim = int.from_bytes(_bytes[2:4])

        requires_grad = bool.from_bytes(_bytes[4:5])

        i = 5 + output_dim * input_dim * 8

        weights = np.frombuffer(_bytes[5:i]).reshape(output_dim)
        biases = np.frombuffer(_bytes[i:])

        layer = Layer.__new__(cls)

        layer.weights = Tensor(weights, requires_grad)
        layer.biases = Tensor(biases, requires_grad)

        return layer



class Network:
    __slots__ = ('layers', 'name')

    def __init__(self, *layers: Layer, name: str = '') -> None:
        self.layers = tuple(layers)
        self.name = name

    @property
    def parameters(self) -> tuple[Tensor, ...]:
        return tuple(parameter for layer in self.layers for parameter in layer.parameters)

    def forward(self, input: Tensor) -> Tensor: ...

    def save(self, save_path: str) -> None:
        layers: list[bytes] = []

        for layer in self.layers:
            layer_data = layer.get_save_data()
            layer_bytes = len(layer_data).to_bytes(4)

            layers.append(layer_bytes + layer_data)

        name = self.name.encode()
        name_length = len(name).to_bytes(2)

        with open(save_path, 'wb') as f:
            f.write(
                name_length + 
                name + 
                b''.join(layers)
            )

    @classmethod
    def from_file(cls, filename: str) -> 'Network':
        layers: list[Layer] = []

        with open(filename, 'rb') as f:
            # get name
            name_length = int.from_bytes(f.read(2))
            name = f.read(name_length).decode()

            # get layers
            while layer_bytes := int.from_bytes(f.read(4)):
                layer = Layer.from_bytes(f.read(layer_bytes))

                layers.append(layer)

        # construct network
        network = cls(*layers, name=name)

        return network

