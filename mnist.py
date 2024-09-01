import struct
import numpy as np

from random import choice, shuffle
from neural_network.tensor import Tensor
from neural_network.data_point import DataPoint



class MNIST:
    __slots__ = ('training_images', 'test_images')

    def __init__(self):
        with open('C:/Users/joemh/Programming/Python/Modules/neural_network/mnist_data/train-images.idx3-ubyte', 'rb') as f:
            f.read(4)

            img_count: int = struct.unpack('>i', f.read(4))[0]
            img_size: int = struct.unpack('>i', f.read(4))[0] * struct.unpack('>i', f.read(4))[0]

            img_values: int = img_count * img_size

            img_data: list[float] = list(struct.unpack('>' + 'B' * img_values, f.read(img_values)))

        with open('C:/Users/joemh/Programming/Python/Modules/neural_network/mnist_data/train-labels.idx1-ubyte', 'rb') as f:
            f.read(4)

            lbl_count: int = struct.unpack('>i', f.read(4))[0]
            lbl_data: list[int] = list(struct.unpack('>' + 'B' * lbl_count, f.read(lbl_count)))

        self.training_images: list[DataPoint] = []

        for i in range(min(img_count, lbl_count)):
            data = Tensor(np.array(img_data[i * img_size:(i + 1) * img_size]))

            label = Tensor(np.zeros(10))
            label[lbl_data[i]] = 1

            new_img = DataPoint(data, label)

            self.training_images.append(new_img)

        with open('C:/Users/joemh/Programming/Python/Modules/neural_network/mnist_data/test-images.idx3-ubyte', 'rb') as f:
            f.read(4)

            img_count: int = struct.unpack('>i', f.read(4))[0]
            img_size: int = struct.unpack('>i', f.read(4))[0] * struct.unpack('>i', f.read(4))[0]
    
            img_values: int = img_count * img_size

            img_data: list[float] = list(struct.unpack('>' + 'B' * img_values, f.read(img_values)))

        with open('C:/Users/joemh/Programming/Python/Modules/neural_network/mnist_data/test-labels.idx1-ubyte', 'rb') as f:
            f.read(4)

            lbl_count: int = struct.unpack('>i', f.read(4))[0]
            lbl_data: list[int] = list(struct.unpack('>' + 'B' * lbl_count, f.read(lbl_count)))

        self.test_images: list[DataPoint] = []

        for i in range(min(img_count, lbl_count)):
            data = Tensor(np.array(img_data[i * img_size:(i + 1) * img_size]))

            label = Tensor(np.zeros(10))
            label[lbl_data[i]] = 1

            new_img = DataPoint(data, label)

            self.test_images.append(new_img)

    def get_random_training_image(self) -> DataPoint:
        return choice(self.training_images)

    def get_random_test_image(self) -> DataPoint:
        return choice(self.test_images)

    def get_random_training_images(self, count: int) -> list[DataPoint]:
        shuffle(self.training_images)
        return self.training_images[:count]

    def get_random_test_images(self, count: int) -> list[DataPoint]:
        shuffle(self.test_images)
        return self.test_images[:count]

