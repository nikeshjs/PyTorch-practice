# PyTorch fundamentals

import torch

#Introduction to Tensors
#Tensors are way to represent data (numeric data)

#scalar
scalar = torch.tensor(7)

#PyTorch tensors are created using torch.tensor().
print(scalar.ndim) #Dimension of a tensor
print(scalar.item()) #Get tensor back as python 'int'

#Vectors
vector = torch.tensor([7,7])
print(vector)
#If you run vector.item(), there will be an error saying "A tensor with 2 elements cannot be converted to scalar."
print("Dimension:", vector.ndim)
print(vector.shape)

#Matrix
matrix = torch.tensor([[1,2],
                      [3,4]])

print(matrix[1])
print(matrix.ndim)
print(matrix.shape) # torch.Size([2,2])

#TENSOR
TENSOR = torch.tensor([[1,2,3],
                        [4,5,6],
                        [7,8,9]])

print(TENSOR)
print(TENSOR.ndim, TENSOR.shape)  #torch.Size([3,3])

# In this case, output will be tensor.Size([1,3,3])
TENSOR = torch.tensor([[[1,2,3],
                        [4,5,6],
                        [7,8,9]]])

print(TENSOR)
print(TENSOR.ndim, TENSOR.shape)  

TENSOR = torch.tensor([[[[1,2,3],
                        [4,5,6],
                        [7,8,9]]], [[[1,2,3],
                        [4,5,6],
                        [7,8,9]]]])
 
print(TENSOR)
print(TENSOR.ndim, TENSOR.shape)  

#RANDOM TENSORS
#Random tensors are important because the way many neural networks learn is that they start with tensors full of random numbers and then adjust those random numbers to better represent the data.
#Random numbers --> look at data --> update random numbers --> look at data --> update random numbers

#Create a random tensor of size (3,4)
random_tensor = torch.rand(3,4)
print(random_tensor)
