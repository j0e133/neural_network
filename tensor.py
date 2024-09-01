import numpy as np

from typing import Sequence, Callable, Any



np.set_printoptions(suppress=True) # stop numpy scientific notation


Array = np.ndarray[float, Any]
GradFunc = Callable[[Array], Sequence[Array]]

_no_grad = False



class _Nograd:
    def __init__(self) -> None:
        pass

    def __enter__(self) -> '_Nograd':
        global _no_grad
        _no_grad = True
        return self
    
    def __exit__(self, *_) -> None:
        global _no_grad
        _no_grad = False



def no_grad() -> _Nograd:
    '''
    Use `with nograd():` to not calculate gradients on any tensor operations under the with.
    '''
    return _Nograd()



def tensor_operation(func: Callable[..., 'Tensor']):
    '''
    Use as a decorator for a function to make it work with tehsor backpropagation.

    The function you use it on can have typehinting that you want, but should return the output of the function, and the backpropagation function instead of an actual output.
    '''
    def apply(*args: 'TensorDataType') -> 'Tensor':
        args = tuple(arg if isinstance(arg, Tensor) else Tensor(arg) for arg in args)
        requires_grad = False if _no_grad else any(arg.requires_grad for arg in args)

        data: Array
        grad_func: GradFunc
        data, grad_func = func(tuple(arg.data for arg in args)) # type: ignore

        return Tensor(
            data,
            requires_grad,
            grad_func if requires_grad else None,
            args if requires_grad else None
        )

    return apply



class Tensor:
    __slots__ = ('data', 'size', 'shape', 'requires_grad', 'grad', '_grad_func', '_prev_tensors')

    def __init__(self, data: Array | float | Sequence[float | Sequence[float]], requires_grad: bool = False, grad_func: GradFunc | None = None, prev_tensors: 'Sequence[Tensor] | None' = None):
        self.data = np.array(data, dtype=float)
        
        if self.data.ndim == 0:
            self.data = np.array([self.data.item()])

        self.size = self.data.size
        self.shape = self.data.shape

        self.requires_grad = requires_grad
        self.grad = None

        self._grad_func = grad_func
        self._prev_tensors = None if prev_tensors is None else tuple(prev_tensors)

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = np.zeros(self.shape)

    def backpropagate(self, grad: None | Array = None) -> None:
        if grad is None:
            if self.size != 1:
                raise ValueError('Gradient must be specified for non-scalar outputs.')

            grad = np.ones(self.shape)

        self.grad += grad

        if self._grad_func is not None and self._prev_tensors is not None:
            grads = self._grad_func(grad)

            for tensor, g in zip(self._prev_tensors, grads):
                if tensor.requires_grad:
                    tensor.backpropagate(g)

    def __repr__(self) -> str:
        return f'Tensor({self.data}, requires_grad={self.requires_grad})'

    def __eq__(self, other: 'TensorDataType') -> bool:
        if isinstance(other, Tensor):
            return np.all(self.data == other.data) # type: ignore

        else:
            return self.data == other

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Array:
        return self.data[index]

    def __setitem__(self, index: int, value: float) -> None:
        self.data[index] = value

    @tensor_operation
    def __neg__(self) -> 'Tensor':
        def back(grad: Array) -> Sequence[Array]:
            return [-grad]

        return (
            -self.data,
            back
        ) # type: ignore

    @tensor_operation
    def __add__(self, other: 'Tensor') -> 'Tensor':
        def back(grad: Array) -> Sequence[Array]:
            return grad, grad

        return (
            self.data + other.data,
            back
        ) # type: ignore

    @tensor_operation
    def __sub__(self, other: 'Tensor') -> 'Tensor':
        def back(grad: Array) -> Sequence[Array]:
            return grad, -grad

        return (
            self.data - other.data,
            back
        ) # type: ignore

    @tensor_operation
    def __mul__(self, other: 'Tensor') -> 'Tensor':
        def back(grad: Array) -> Sequence[Array]:
            return grad * other.data, grad * self.data

        return (
            self.data * other.data,
            back
        ) # type: ignore

    @tensor_operation
    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        def back(grad: Array) -> Sequence[Array]:
            return (grad / other.data, -grad * self.data / np.square(other.data))

        return (
            self.data / other.data,
            back
        ) # type: ignore

    @tensor_operation
    def __pow__(self, other: float) -> 'Tensor':
        def back(grad: Array) -> Sequence[Array]:
            return [grad * other * (self.data ** (other - 1))]

        return (
            self.data ** other,
            back
        ) # type: ignore

    @tensor_operation
    def sum(self) -> 'Tensor':
        '''
        Calculates the sum of this tensor's elements.
        '''

        def back(grad: Array) -> Sequence[Array]:
            return [grad * np.ones_like(self.data)]

        return (
            self.data.sum(),
            back
        ) # type: ignore

    @tensor_operation
    def dot(self, other: 'Tensor') -> 'Tensor':
        '''
        Calculates the dot product of this tensor with another one.
        '''

        def back(grad: Array) -> Sequence[Array]:
            return ((np.outer(grad, other.data), self.data.T.dot(grad)) if self.data.ndim > 1 else (grad * other.data, grad * self.data))

        return (
            self.data.dot(other.data),
            back
        ) # type: ignore

    @tensor_operation
    def mean(self) -> 'Tensor':
        '''
        Calculates the mean of this tensor's elements.
        '''

        def back(grad: Array) -> Sequence[Array]:
            return [grad * np.ones_like(self.data) / len(self.data)]

        return (
            self.data.mean(),
            back
        ) # type: ignore

    def copy(self, requires_grad: bool | None = None) -> 'Tensor':
        return Tensor(
            self.data,
            self.requires_grad if requires_grad is None else requires_grad
        )

    @tensor_operation
    def onehot(self) -> 'Tensor':
        '''
        Returns a tensor with the index of the max value equal to 1 and everything else equal to 0.
        '''

        argmax = np.argmax(self.data)

        data = np.zeros_like(self.data)
        data[argmax] = 1

        def back(grad: Array) -> Sequence[Array]:
            return [grad * data]

        return (
            data,
            back
        ) # type: ignore



@tensor_operation
def square(tensor: Tensor) -> 'Tensor':
    '''
    Calculates the element-wise square of a tensor's elements

    Much faster than `Tensor() ** 2`
    '''

    def back(grad: Array) -> Sequence[Array]:
        return [grad * 2 * tensor.data]

    return (
        np.square(tensor.data),
        back
    ) # type: ignore


@tensor_operation
def sqrt(tensor: Tensor) -> 'Tensor':
    '''
    Calculates the element-wise square root of a tensor's elements

    Much faster than `Tensor() ** 0.5`
    '''

    _sqrt = np.sqrt(tensor.data)

    def back(grad: Array) -> Sequence[Array]:
        return [grad * 0.5 / _sqrt]

    return (
        _sqrt,
        back
    ) # type: ignore


@tensor_operation
def exp(tensor: Tensor) -> 'Tensor':
    '''
    Calculates the element-wise exponential of a tensor's elements.
    '''

    _exp = np.exp(tensor.data)

    def back(grad: Array) -> Sequence[Array]:
        return [grad * _exp]

    return (
        _exp,
        back
    ) # type: ignore


@tensor_operation
def log(tensor: Tensor) -> 'Tensor':
    '''
    Calculates the element-wise logarithm of a tensor's elements.
    '''

    def back(grad: Array) -> Sequence[Array]:
        return [grad / tensor.data]

    return (
        np.log(tensor.data),
        back
    ) # type: ignore


@tensor_operation
def dot(t1: Tensor, t2: Tensor) -> Tensor:
    '''
    Calculates the dot product of two tensors.
    '''

    def back(grad: Array) -> Sequence[Array]:
        return ((np.outer(grad, t2.data), t1.data.T.dot(grad)) if t1.data.ndim > 1 else (grad * t2.data, grad * t1.data))

    return (
        t1.data.dot(t2.data),
        back
    ) # type: ignore


@tensor_operation
def min(t1: Tensor, t2: Tensor) -> Tensor:
    '''
    Calculates the element-wise min of two tensors.
    '''

    def back(grad: Array) -> Sequence[Array]:
        mask = (t1.data <= t2.data)
        return (grad * mask, grad * ~mask)

    return (
        np.minimum(t1.data, t2.data),
        back
    ) # type: ignore


@tensor_operation
def max(t1: Tensor, t2: Tensor) -> Tensor:
    '''
    Calculates the element-wise max of two tensors.
    '''

    def back(grad: Array) -> Sequence[Array]:
        mask = (t1.data > t2.data)
        return (grad * mask, grad * ~mask)

    return (
        np.maximum(t1.data, t2.data),
        back
    ) # type: ignore


@tensor_operation
def concatenate(t1: Tensor, t2: Tensor) -> Tensor:
    '''
    Concatenates two tensors.
    '''

    def back(grad: Array) -> Sequence[Array]:
        return np.split(grad, t1.shape)

    return (
        np.concatenate((t1.data, t2.data)),
        back
    ) # type: ignore
    


TensorDataType = Tensor | Array | float

