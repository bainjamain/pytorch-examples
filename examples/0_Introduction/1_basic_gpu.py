import torch
import numpy as np
import time

# Let's compare the running time of a simple operation with vanilla NumPy
# and its counterpart PyTorch on GPU.
n = 10000
A = np.random.randn(n, n)

start = time.time()
A2 = np.matmul(A, A)
print('NumPy A ** 2: %.4f seconds' % (time.time() - start)) # 7.4536 seconds

A = torch.from_numpy(A).cuda() # puting a tensor to a GPU
start = time.time()
A2 = torch.mm(A, A)
print('PyTorch with GPU A ** 2: %.4f seconds' % (time.time() - start)) # 0.2152 second

# On this simple example, we note a 30x speed up
# in the matrix multiplication with a Titan X GPU.

# ---------

# Let's look more closely how to map a calculation to
# a specific GPU (in case of a multi-GPU system)
print('Number of GPUs: %i' % torch.cuda.device_count()) # Output: 4
print('ID of the GPU used: %i' % torch.cuda.current_device()) # current default GPU. Output: 0
torch.cuda.set_device(1) # switch to GPU 1
print('ID of the GPU used: %i' % torch.cuda.current_device()) # Output: 1

# Using context manager to place operations on a given device
with torch.cuda.device(0):
    A = torch.randn(n, n).cuda()
    A2 = A.mm(A)
print('A is on GPU %i' % (A.get_device())) # Output: 0
      
with torch.cuda.device(3):
    A = torch.randn(n, n).cuda()
    A2 = A.mm(A)
print('A is on GPU %i' % (A.get_device())) # Output: 3
