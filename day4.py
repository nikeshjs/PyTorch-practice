# Selecting data from tensors
# Similar to indexing with NumPy

import torch
import numpy as np
x = torch.arange(1,10).reshape(1,3,3)
print(x,x[0][1][2])
print(x[:,:,2])


#PyTorch tensors and NumPy
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array, "\n", tensor)