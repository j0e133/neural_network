import numpy as np

from neural_network.tensor import Tensor



class DataPoint:
    __slots__ = ('input', 'expected_output')

    def __init__(self, input: Tensor, expected_output: Tensor) -> None:
        self.input = input
        self.expected_output = expected_output

    def noisy_input(self, noise_magnitude: float) -> Tensor:
        return self.input + np.random.uniform(-noise_magnitude, noise_magnitude, self.input.shape)
    
