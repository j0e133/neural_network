import numpy as np

from neural_network.tensor import Tensor



EPSILON = 1e-8



class Optimizer:
    __slots__ = ('parameters')

    def __init__(self, *parameters: Tensor) -> None:
        self.parameters = parameters

    def apply_gradients(self, learning_rate: float) -> None: ...

    def zero_grad(self) -> None:
        '''
        Clear the gradients of all parameters

        is automatically done after `Optimizer.apply_gradients()` is called
        '''

        for parameter in self.parameters:
            parameter.zero_grad()



class GradientDescent(Optimizer):
    def apply_gradients(self, learning_rate: float) -> None:
        for parameter in self.parameters:
            # apply gradient
            parameter.data -= learning_rate * parameter.grad # type: ignore

            # clear gradient
            parameter.zero_grad()



class MomentumGradientDescent(Optimizer):
    __slots__ = ('momentum', 'beta')

    def __init__(self, *parameters: Tensor, beta: float = 0.9) -> None:
        super().__init__(*parameters)

        self.momentum = tuple(np.zeros_like(parameter.grad) for parameter in parameters)

        self.beta = beta

    def apply_gradients(self, learning_rate: float) -> None:
        for parameter, momentum in zip(self.parameters, self.momentum):
            # update momentum
            momentum *= self.beta
            momentum += (1 - self.beta) * parameter.grad # type: ignore

            # apply gradient
            parameter.data -= learning_rate * momentum # type: ignore

            # clear gradient
            parameter.zero_grad()



class RMSProp(Optimizer):
    '''
    Root Mean Squared Propagation
    '''

    __slots__ = ('variance', 'beta')

    def __init__(self, *parameters: Tensor, beta: float = 0.99) -> None:
        super().__init__(*parameters)

        self.variance = tuple(np.zeros_like(parameter.grad) for parameter in parameters)

        self.beta = beta

    def apply_gradients(self, learning_rate: float) -> None:
        for parameter, variance in zip(self.parameters, self.variance):
            # update variance
            variance *= self.beta
            variance += (1 - self.beta) * np.square(parameter.grad) # type: ignore

            # apply gradient
            parameter.data -= learning_rate * parameter.grad / np.sqrt(variance + EPSILON) # type: ignore

            # clear gradient
            parameter.zero_grad()



class ADAM(Optimizer):
    '''
    Adaptive Moments
    '''

    __slots__ = ('momentum', 'variance', 'beta', 'gamma', '_i')

    def __init__(self, *parameters: Tensor, beta: float = 0.9, gamma: float = 0.995) -> None:
        super().__init__(*parameters)

        self.momentum = tuple(np.zeros_like(parameter.grad) for parameter in parameters)
        self.variance = tuple(np.zeros_like(parameter.grad) for parameter in parameters)

        self.beta = beta
        self.gamma = gamma

        self._i = 0

    def apply_gradients(self, learning_rate: float) -> None:
        self._i += 1

        beta_prime = 1 / (1 - self.beta ** self._i)
        gamma_prime = 1 / (1 - self.gamma ** self._i)

        for parameter, momentum, variance in zip(self.parameters, self.momentum, self.variance):
            # update momentum
            momentum *= self.beta
            momentum += (1 - self.beta) * parameter.grad # type: ignore

            # update variance
            variance *= self.gamma
            variance += (1 - self.gamma) * np.square(parameter.grad) # type: ignore
            
            # calculate time dialated values
            m_hat = momentum * beta_prime
            v_hat = variance * gamma_prime

            # apply gradient
            parameter.data -= learning_rate * m_hat / (np.sqrt(v_hat) + EPSILON)

            # clear gradient
            parameter.zero_grad()

