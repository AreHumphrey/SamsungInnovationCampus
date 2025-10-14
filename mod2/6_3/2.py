
import torch
import torch.nn as nn

input_size = 7
batch_size = 5
input_tensor = torch.randn(batch_size, input_size, dtype=torch.float)

eps = 1e-3

def custom_batch_norm1d(input_tensor, weight, bias, eps):
    mean = input_tensor.mean(dim=0, keepdim=True)
    var = ((input_tensor - mean) ** 2).mean(dim=0, keepdim=True)
    normed_tensor = (input_tensor - mean) / torch.sqrt(var + eps)
    normed_tensor = normed_tensor * weight + bias
    return normed_tensor

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# batch_norm = nn.BatchNorm1d(input_size, eps=eps)
# batch_norm.bias.data = torch.randn(input_size, dtype=torch.float)
# batch_norm.weight.data = torch.randn(input_size, dtype=torch.float)
# batch_norm_out = batch_norm(input_tensor)
# custom_batch_norm_out = custom_batch_norm1d(input_tensor, batch_norm.weight.data, batch_norm.bias.data, eps)
# print(torch.allclose(batch_norm_out, custom_batch_norm_out, 1e-3) \
#       and batch_norm_out.shape == custom_batch_norm_out.shape)
# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# batch_norm = nn.BatchNorm1d(input_size, eps=eps)
# batch_norm.bias.data = torch.randn(input_size, dtype=torch.float)
# batch_norm.weight.data = torch.randn(input_size, dtype=torch.float)
# batch_norm_out = batch_norm(input_tensor)
# custom_batch_norm_out = custom_batch_norm1d(input_tensor, batch_norm.weight.data, batch_norm.bias.data, eps)
# print(torch.allclose(batch_norm_out, custom_batch_norm_out, 1e-3) \
#       and batch_norm_out.shape == custom_batch_norm_out.shape)