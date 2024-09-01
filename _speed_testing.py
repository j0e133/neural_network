import numpy as np

from timeit import timeit



t1 = np.random.uniform(-1, 1, 100)
t2 = np.random.uniform(-1, 1, 100)
diff = t1 - t2

print(np.square(diff).mean())
print(diff.dot(diff) / len(diff))

raise Exception()



setup = '''
t1 = np.random.uniform(-1, 1, 100)
t2 = np.random.uniform(-1, 1, 100)
diff = t1 - t2
# grad = np.random.uniform(-10, 10, 100)
'''

globals = {
    'np': np
}

a = '''
np.square(diff).mean()
'''

b = '''
diff.dot(diff) / len(diff)
'''

print(timeit(a, setup, globals=globals))
print(timeit(b, setup, globals=globals))

