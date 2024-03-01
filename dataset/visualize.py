
import numpy as np
import time

start = time.time()

a = np.random.rand(1000, 64)
b = np.random.rand(64, 10000)
times = 100
results = []
for i in range(times):
    # a_inv = np.linalg.pinv(a)
    y = np.matmul(a, b)
    results += [y]

end = time.time()

print(f'Time (ms) = {(end - start) * 1000 / times}')
