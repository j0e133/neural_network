import numpy as np

from neural_network.tensor import Array



EPSILON = 1e-8



class Optimizer:
    def __init__(self, weights: list[Array], biases: list[Array]) -> None: ...

    def __call__(self, layer: int, weight_gradient: Array, bias_gradient: Array) -> tuple[Array, Array]: ... # type: ignore



class GradientDescent(Optimizer):
    def __call__(self, layer: int, weight_gradient: Array, bias_gradient: Array) -> tuple[Array, Array]:
        return weight_gradient, bias_gradient



class MomentumGradientDescent(Optimizer):
    __slots__ = ('weight_momentum', 'bias_momentum', 'beta')

    def __init__(self, weights: list[Array], biases: list[Array], beta: float = 0.9) -> None:
        super().__init__(weights, biases)

        self.weight_momentum = [np.zeros(layer.shape) for layer in weights]
        self.bias_momentum = [np.zeros(layer.shape) for layer in biases]

        self.beta = beta

    def __call__(self, layer: int, weight_gradient: Array, bias_gradient: Array) -> tuple[Array, Array]:
        # weights
        self.weight_momentum[layer] *= self.beta
        self.weight_momentum[layer] += (1 - self.beta) * weight_gradient

        # biases
        self.bias_momentum[layer] *= self.beta
        self.bias_momentum[layer] += (1 - self.beta) * bias_gradient

        return self.weight_momentum[layer], self.bias_momentum[layer]



class RMSProp(Optimizer):
    '''
    Root Mean Squared Propagation
    '''

    __slots__ = ('weight_variance', 'bias_variance', 'beta')

    def __init__(self, weights: list[Array], biases: list[Array], beta: float = 0.9) -> None:
        super().__init__(weights, biases)

        self.weight_variance = [np.zeros(layer.shape) for layer in weights]
        self.bias_variance = [np.zeros(layer.shape) for layer in biases]

        self.beta = beta

    def __call__(self, layer: int, weight_gradient: Array, bias_gradient: Array) -> tuple[Array, Array]:
        # weights
        self.weight_variance[layer] *= self.beta
        self.weight_variance[layer] += (1 - self.beta) * np.square(weight_gradient)

        # biases
        self.bias_variance[layer] *= self.beta
        self.bias_variance[layer] += (1 - self.beta) * np.square(bias_gradient)

        return weight_gradient / np.sqrt(self.weight_variance[layer] + EPSILON), bias_gradient / np.sqrt(self.bias_variance[layer] + EPSILON)



class ADAM(Optimizer):
    '''
    Adaptive Moments
    '''

    __slots__ = ('weight_momentum', 'bias_momentum', 'weight_variance', 'bias_variance', 'beta', 'gamma', '_i')

    def __init__(self, weights: list[Array], biases: list[Array], *, beta: float = 0.9, gamma: float = 0.99) -> None:
        super().__init__(weights, biases)

        self.weight_momentum = [np.zeros(layer.shape) for layer in weights]
        self.bias_momentum = [np.zeros(layer.shape) for layer in biases]

        self.weight_variance = [np.zeros(layer.shape) for layer in weights]
        self.bias_variance = [np.zeros(layer.shape) for layer in biases]

        self.beta = beta
        self.gamma = gamma

        self._i = 0

    def __call__(self, layer: int, weight_gradient: Array, bias_gradient: Array) -> tuple[Array, Array]:
        self._i += 1

        beta_prime = 1 / (1 - self.beta ** self._i)
        gamma_prime = 1 / (1 - self.gamma ** self._i)

        # weights
        self.weight_momentum[layer] *= self.beta
        self.weight_momentum[layer] += (1 - self.beta) * weight_gradient

        self.weight_variance[layer] *= self.gamma
        self.weight_variance[layer] += (1 - self.gamma) * np.square(weight_gradient)

        weight_m_hat = self.weight_momentum[layer] * beta_prime
        weight_v_hat = self.weight_variance[layer] * gamma_prime

        # biases
        self.bias_momentum[layer] *= self.beta
        self.bias_momentum[layer] += (1 - self.beta) * bias_gradient

        self.bias_variance[layer] *= self.gamma
        self.bias_variance[layer] += (1 - self.gamma) * np.square(bias_gradient)

        bias_m_hat = self.bias_momentum[layer] * beta_prime
        bias_v_hat = self.bias_variance[layer] * gamma_prime

        return weight_m_hat / (np.sqrt(weight_v_hat) + EPSILON), bias_m_hat / (np.sqrt(bias_v_hat) + EPSILON)

