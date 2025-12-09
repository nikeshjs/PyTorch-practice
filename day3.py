# MATRIX MULTIPLICATION IN PYTORCH
import torch
tensor_one = torch.tensor([1,2,3])
print(torch.matmul(tensor_one, tensor_one))
# Also:
print(tensor_one @ tensor_one)

tensor_two = torch.tensor([[1,2],
                           [3,4],
                           [5,6]])
print(tensor_two.shape) #Just testing

print(tensor_two.T) #Transpose of the tensor_two


# Finding the min, max, mean, sum etc (tensor aggregation)
tensor_three = torch.tensor([11,234,56,65,78])
print(torch.min(tensor_three), ",", torch.max(tensor_three))
# For mean(), input dtype should be either floating point or complex dtype. It cannot work on long (int64).
print(torch.mean(tensor_three.type(torch.float32)))
# Also, tensor_three.min(), tensor_three.max(), tensor_three.type(torch.float32).mean()
print(torch.sum(tensor_three))
print(torch.argmax(tensor_three), torch.argmin(tensor_three)) #Printing the index of the occured min and max values


# Reshaping, stacking, squeezing and unsqueezing tensors
'''
Reshaping - reshapes an input tensor to defined shape
view - return a view of an input tensor of certain shape but keep the same memory as the original tensor
stacking - combining multiple tensors on top of each other (vstack) or side by side (hstack)
squeeze - removes all '1' dimensions from a tensor
unsqueeze - add a '1' dimension to a target tensor
permute - returna a view of the input with dimensions permuted (swapped) in a certain way
'''
tensor_four = torch.arange(0,20,1)
print(tensor_four)
#add an extra dimension
tensor_reshaped = tensor_four.reshape(2,10)
print(tensor_reshaped)

# Change the view
tensor_four_view = tensor_four.view(2,10)
# Changes in tensor_four_view changes the original tensor_four too because they share the same memory

# tensor_stacked = torch.stack([tensor_one, tensor_two, tensor_three, tensor_four])
# Got an error here that said stack expects each tensor to be of equal size. Here, all the tensor sizes are different

tensor_squeezed = tensor_reshaped.squeeze() # or torch.squeeze(tensor_reshaped)
print(tensor_squeezed, tensor_squeezed.shape)

# Similar for torch.unsqueeze()


# torch.permute - rearranges the dimensions of a target tensor in a specified order
tensor_five = torch.rand(5,2,3)
tensor_permuted = torch.permute(tensor_five, (2,0,1)) #(2,0,1) is the index of the size of the original tensor (tensor_five)
print(tensor_permuted, tensor_permuted.shape)

# Timestamp: 3:23:50