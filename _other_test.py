import numpy as np



w = np.array(
    (
        (0, 1, 2, 3),
        (4, 5, 6, 7)
    ),
    float
)
b = np.array(
    (8, 9),
    float
)

shape_bytes = bytes(w.shape)
grad_bytes = int(True).to_bytes(2)
wbytes = w.tobytes()
bbytes = b.tobytes()

size = len(shape_bytes) + len(grad_bytes) + len(wbytes) + len(bbytes)
sizebytes = size.to_bytes(4)

out = sizebytes + shape_bytes + grad_bytes + wbytes + bbytes

print(size)
print(len(sizebytes))
print(len(out))



size = int.from_bytes(out[:1])
_bytes = out[4:]

shape = (int.from_bytes(_bytes[:1]), int.from_bytes(_bytes[1:2]))
req_grad = bool.from_bytes(_bytes[2:4])
i = 4 + shape[0] * shape[1] * 8
weights = np.frombuffer(_bytes[4:i]).reshape(shape)
biases = np.frombuffer(_bytes[i:])

print(shape)
print(req_grad)
print(weights)
print(biases)

