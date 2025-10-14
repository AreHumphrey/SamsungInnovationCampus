import torch
import torch.nn as nn

eps = 1e-3

input_channels = 3
batch_size = 3
height = 10
width = 10

batch_norm_2d = nn.BatchNorm2d(input_channels, affine=False, eps=eps)

input_tensor = torch.randn(batch_size, input_channels, height, width, dtype=torch.float)


def custom_batch_norm2d(input_tensor, eps):

    mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)
    var = input_tensor.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

    normed_tensor = (input_tensor - mean) / torch.sqrt(var + eps)

    return normed_tensor

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# norm_output = batch_norm_2d(input_tensor)
# custom_output = custom_batch_norm2d(input_tensor, eps)
# print(torch.allclose(norm_output, custom_output) and norm_output.shape == custom_output.shape)
