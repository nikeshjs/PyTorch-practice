import torch
# Pytorch reproducibility -> Trying to take random out of random
# How a neural network learns: Start with random numbers, tensor operations, update random numbers to try and make them better representations of the data, repeat

# To reduce the randomness in neural networks and PyTorch, there is a concept of random seed. What it does is it "flavours" the randomness.

tensor_a = torch.rand(3,4)
tensor_b = torch.rand(3,4)
print(tensor_a, tensor_b, tensor_a == tensor_b)
#Random but reproducible tensors
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
tensor_c = torch.rand(3,4) #Flavours the tensor

# Timestamp = 4:17:40
