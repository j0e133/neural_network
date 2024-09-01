import numpy as np

from typing import Callable, Sequence
from tensor import Tensor, tensor_operation, Array



def _clipped_negative_exp(array: Array) -> Array:
    return np.exp(-np.clip(array, -700, 700))



########################
# Activation Functions #
########################

@tensor_operation
def ReLU(tensor: Tensor) -> Tensor:
    '''
    The Rectified Linear Unit function, equivalent to elementwise `max(x, 0)`
    
    Best for hidden and input layers
    '''

    def back(grad: Array) -> Sequence[Array]:
        return [grad * (tensor.data > 0)]

    return (
        np.maximum(tensor.data, 0),
        back
    ) # type: ignore


@tensor_operation
def sigmoid(tensor: Tensor) -> Tensor:
    '''
    The sigmoid function maps all values of x to the range (0, 1)

    Best for output layers
    '''

    def back(grad: Array) -> Sequence[Array]:
        exp = _clipped_negative_exp(tensor.data)
        return [grad * exp / np.square(1 + exp)]

    return (
        1 / (1 + _clipped_negative_exp(tensor.data)),
        back
    ) # type: ignore


@tensor_operation
def SiLU(tensor: Tensor) -> Tensor:
    '''
    The Sigmoid Linear Unit is a smooth version of ReLU which can have negative values

    Best for hidden and input layers
    '''

    def back(grad: Array) -> Sequence[Array]:
        sigmoid_ = 1 / (1 + _clipped_negative_exp(tensor.data))
        return [grad * sigmoid_ * (1 + tensor.data * (1 - sigmoid_))]

    return (
        tensor.data / (1 + _clipped_negative_exp(tensor.data)),
        back
    ) # type: ignore


@tensor_operation
def tanh(tensor: Tensor) -> Tensor:
    '''
    Tanh maps all values of x to the range (-1, 1)

    Best for hidden and input layers
    '''

    _tanh = np.tanh(tensor.data)

    def back(grad: Array) -> Sequence[Array]:
        return [grad * np.square(1 - _tanh)]

    return (
        _tanh,
        back
    ) # type: ignore


@tensor_operation
def arcsinh(tensor: Tensor) -> Tensor:
    '''
    Arcsinh maps all values of x so that x increases/decreases very slowly the farther it gets from the y axis

    Best for hidden and input layers
    '''

    def back(grad: Array) -> Sequence[Array]:
        return [grad / np.sqrt(1 + np.square(tensor.data))]

    return (
        np.arcsinh(tensor.data),
        back
    ) # type: ignore


@tensor_operation
def softmax(tensor: Tensor) -> Tensor:
    '''
    Softmax maps all values in an array so they sum to 1

    Works best in turning an unbounded set of numbers into probabilities
    '''

    exp_array: Array = np.exp(tensor.data - tensor.data.max())

    def back(grad: Array) -> Sequence[Array]:
        return [grad]

    return (
        exp_array / exp_array.sum(),
        back
    ) # type: ignore


##################
# Loss Functions #
##################

@tensor_operation
def cross_entropy(observed: Tensor, expected: Tensor) -> Tensor:
    '''
    Cross Entropy measures how different the observed values are from the expected
    '''

    def back(grad: Array) -> Sequence[Array]:
        return [grad * (observed.data - expected.data)]

    return (
        -(expected.data * np.log(observed.data)).mean(),
        back
    ) # type: ignore


@tensor_operation
def mean_squared_error(observed: Tensor, expected: Tensor) -> Tensor:
    '''
    Mean Squared Error calculates the mean of the square difference between the observed and expected values

    Error increases faster the bigger the difference is compared to Mean Absolute Error
    '''

    diff = (observed.data - expected.data)

    def back(grad: Array) -> Sequence[Array]:
        return [grad * (2 / observed.shape[0]) * diff]

    return (
        diff.dot(diff) / len(diff),
        back
    ) # type: ignore


@tensor_operation
def mean_absolute_error(observed: Tensor, expected: Tensor) -> Tensor:
    '''
    Mean Absolute Error calculates the mean of the difference between the observed and expected values
    '''

    def back(grad: Array) -> Sequence[Array]:
        derivative = np.where(observed.data > expected.data, 1, -1)
        derivative[observed == expected] = 0

        return [derivative]

    return (
        np.abs(expected.data - observed.data).mean(),
        back
    ) # type: ignore


#############
# Save maps #
#############

AFUNCS: dict[str | Callable[[Tensor], Tensor], str | Callable[[Tensor], Tensor]] = {
    'ReLU':    ReLU,
    'sigmoid': sigmoid,
    'SiLU':    SiLU,
    'tanh':    tanh,
    'arcsinh': arcsinh,
    'softmax': softmax,
     ReLU:    'ReLU',
     sigmoid: 'sigmoid',
     SiLU:    'SiLU',
     tanh:    'tanh',
     arcsinh: 'arcsinh',
     softmax: 'softmax',
}

LFUNCS: dict[str | Callable[[Tensor, Tensor], Tensor], str | Callable[[Tensor, Tensor], Tensor]] = {
    'cross entropy':       cross_entropy,
    'mean squared error':  mean_squared_error,
    'mean absolute error': mean_absolute_error,
     cross_entropy:       'cross entropy',
     mean_squared_error:  'mean squared error',
     mean_absolute_error: 'mean absolute error',
} # type: ignore

