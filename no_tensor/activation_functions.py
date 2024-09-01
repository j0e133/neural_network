import numpy as np

from typing import Callable
from neural_network.tensor import Array



ActivationFunction = Callable[[Array], Array]
ActivationFunctionDerivative = Callable[[Array], Array]

LossFunction = Callable[[Array, Array], float]
LossFunctionDerivative = Callable[[Array, Array], Array]


def _clipped_negative_exp(array: Array) -> Array:
    return np.exp(-np.maximum(array, -709))


def ReLU(array: Array) -> Array:
    '''
    The Rectified Linear Unit function, equivalent to elementwise `max(x, 0)`
    
    Best for hidden and input layers
    '''
    return np.maximum(array, 0)

def d_ReLU(array: Array) -> Array:
    '''
    Derivative of `ReLU(x)`
    '''
    return 1.0 * (array > 0)


def sigmoid(array: Array) -> Array:
    '''
    The sigmoid function maps all values of x to the range (0, 1)

    Best for output layers
    '''
    return 1 / (1 + _clipped_negative_exp(array))

def d_sigmoid(array: Array) -> Array:
    '''
    Derivative of `sigmoid(x)`
    '''
    exp = _clipped_negative_exp(array)
    return exp / np.square(1 + exp)


def SiLU(array: Array) -> Array:
    '''
    The Sigmoid Linear Unit is a smooth version of ReLU which can have negative values

    Best for hidden and input layers
    '''
    return array / (1 + _clipped_negative_exp(array))

def d_SiLU(array: Array) -> Array:
    '''
    Derivative of `SiLU(x)`
    '''
    sigmoid_ = sigmoid(array)
    return sigmoid_ * (1 + array * (1 - sigmoid_))


def tanh(array: Array) -> Array:
    '''
    Tanh maps all values of x to the range (-1, 1)

    Best for hidden and input layers
    '''
    return np.tanh(array)

def d_tanh(array: Array) -> Array:
    '''
    Derivative of `tanh(x)`
    '''
    return np.square(1 - np.tanh(array))


def arcsinh(array: Array) -> Array:
    '''
    Arcsinh maps all values of x so that x increases/decreases very slowly the farther it gets from the y axis

    Best for hidden and input layers
    '''
    return np.arcsinh(array)

def d_arcsinh(array: Array) -> Array:
    '''
    Derivative of `arcsinh(x)`
    '''
    return 1 / np.sqrt(1 + np.square(array))


def linear(array: Array) -> Array:
    '''
    Returns x

    Best for output layers where you don't need values in a certaint bounds
    '''
    return array

def d_linear(array: Array) -> Array:
    '''
    Derivative of `linear(x)`
    '''
    return np.ones(array.shape)



def softmax(array: Array) -> Array:
    exp_array: Array = np.exp(array - array.max())

    return exp_array / exp_array.sum()

def d_softmax(array: Array) -> Array:
    return np.ones(array.shape)


def cross_entropy(observed: Array, expected: Array) -> float:
    return -(expected * np.log(observed)).mean()

def d_cross_entropy(observed: Array, expected: Array) -> Array:
    return observed - expected


def mean_squared_error(observed: Array, expected: Array) -> float:
    return ((observed - expected) ** 2).mean()

def d_mean_squared_error(observed: Array, expected: Array) -> Array:
    return 2 / observed.shape[0] * (observed - expected)


def mean_absolute_error(observed: Array, expected: Array) -> float:
    return np.abs(expected - observed).mean()

def d_mean_absolute_error(observed: Array, expected: Array) -> Array:
    derivative = np.where(observed > expected, 1, -1)
    derivative[observed == expected] = 0

    return derivative


AFUNCS: dict[str, ActivationFunction] = {
    'ReLU':    ReLU,
    'sigmoid': sigmoid,
    'SiLU':    SiLU,
    'tanh':    tanh,
    'arcsinh': arcsinh,
    'linear':  linear,
    'softmax': softmax,
}
AFUNC_DERIVATIVES: dict[str | ActivationFunction, ActivationFunctionDerivative] = {
    ReLU:      d_ReLU,
    sigmoid:   d_sigmoid,
    SiLU:      d_SiLU,
    tanh:      d_tanh,
    arcsinh:   d_arcsinh,
    linear:    d_linear,
    softmax:   d_softmax,
}

LFUNCS: dict[str, LossFunction] = {
    'cross entropy':       cross_entropy,
    'mean squared error':  mean_squared_error,
    'mse':                 mean_squared_error,
    'mean absolute error': mean_absolute_error,
    'mae':                 mean_absolute_error,
}
LFUNC_DERIVATIVES: dict[str | LossFunction, LossFunctionDerivative] = {
    'cross entropy':       d_cross_entropy,
    cross_entropy:         d_cross_entropy,
    'mean squared error':  d_mean_squared_error,
    'mse':                 d_mean_squared_error,
    mean_squared_error:    d_mean_squared_error,
    'mean absolute error': d_mean_absolute_error,
    'mae':                 d_mean_absolute_error,
    mean_absolute_error:   d_mean_absolute_error,
}
