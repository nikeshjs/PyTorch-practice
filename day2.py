import torch
#Create range of tensors and tensor-like
one_to_ten = torch.arange(start=1, end=10, step=1)
print(one_to_ten) #ends in (end-1)

# Creating tensors-like
zeros = torch.zeros_like(input=one_to_ten)
print(zeros)

ones = torch.ones_like(input=one_to_ten)
print(ones)



#Tensors datatypes: Tensor dtypes are one of the big errors you'll run into with pytorch
#1. Tensors not right dtype
#2. Tensors not right shape
#3. Tensors not on the right device

#Float32 tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0])
print(float_32_tensor.dtype) #torch.float32

int_64_tensor = torch.tensor([3, 6, 9])
print(int_64_tensor.dtype) #torch.int64

#IF we specify our tensor values ourselves, datatype will be as per the specified values. Otherwise, if pytorch generates it, default dtype will be float32.

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               device=None,
                               requires_grad=False)
float_16_tensor = float_32_tensor.type(torch.float16) # If we have to change dtype at some point

# device = none, it specifies the default device used for tensor is CPU. If one tensor is in GPU and another in CPU and we try to perform any operations on them, pytorch throws an error saying tensor not on the right device. If we have to operate on GPU, write "device= "cuda""

# requires_grad = whether or not to track gradients with tensor operations

int_32_tensor = torch.tensor([3,6,9], dtype=torch.int32)
print(int_32_tensor.dtype)

print(int_32_tensor * float_32_tensor) #It works! No dtype issue

# Getting information from tensors
# 1st issue: to get dtype from a tensor, can use tensor.dtype
# 2nd issue: to get shape from a tensor, can use tensor.shape
# 3rd issue: to get device from a tensor, can use tensor.device

# find out details about the tensor
print(int_32_tensor.dtype, int_32_tensor.shape, int_32_tensor.device)

random_tensor = torch.rand(2,2, dtype=torch.float16)
print(random_tensor.dtype, random_tensor.size(), random_tensor.device)

# Manipulating tensors: Tensor operations
'''
Includes: 1. Addtion
2. Subtraction
3. Multiplication (element-wise)
4. Division
5. Matrix multiplication
'''

print(int_32_tensor + 10)
print(int_32_tensor * 10)
print(int_32_tensor - 10)

print(torch.mul(int_32_tensor, 10))
print(torch.add(int_32_tensor, 10))
print(torch.sub(int_32_tensor, 10, alpha=2)) #In this case, alpha scales the number to be subtracted. 10*2=20. So, int_32_integer - 20.
print(torch.subtract(int_32_tensor, 10))
