import torch
import torch.nn as nn

eps = 1e-3

batch_size = 5
input_channels = 2
input_length = 30

instance_norm = nn.InstanceNorm1d(input_channels, affine=False, eps=eps)

input_tensor = torch.randn(batch_size, input_channels, input_length, dtype=torch.float)


def custom_instance_norm1d(input_tensor, eps):
    mean = input_tensor.mean(dim=2, keepdim=True)
    var = input_tensor.var(dim=2, unbiased=False, keepdim=True)

    normed_tensor = (input_tensor - mean) / torch.sqrt(var + eps)

    return normed_tensor

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# norm_output = instance_norm(input_tensor)
# custom_output = custom_instance_norm1d(input_tensor, eps)
# print(torch.allclose(norm_output, custom_output, atol=1e-06) and norm_output.shape == custom_output.shape)
