import torch
import numpy as np

# Let's create a 2x2 matrix filled with 2s
A = torch.Tensor(2, 3) # creates a tensor of shape (2, 3)
A.fill_(2.) # fills the tensor with 2s. In PyTorch, operations postfixed by `_` are in place
# Or, simply...
B = 2. * torch.ones(3, 2)

# Some basic matrix operations
C = A + A
D =  A + 3 # adding a matrix with a scalar
A_T =  A.t() # or A.t_() for in place transposition
AB = A.mm(B) # computes A.B (matrix multiplication), equivalent to A @ B in Python 3.5+
A_h = A * A # computes the element-wise matrix multiplication (Hadamard product)

# Applying a function element-wise to a matrix/tensor
f =  lambda x: x * x
fA = f(A)
# Or, simply
A.apply_(lambda x: x * x)

# Casting an array from NumPy to PyTorch...
A = np.ones((2, 3))
A = torch.from_numpy(A)
A = A.numpy() # ... and back to NumPy
