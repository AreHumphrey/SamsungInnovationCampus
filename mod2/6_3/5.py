import torch
import torch.nn as nn

eps = 1e-10


def custom_layer_norm(input_tensor, eps):
    mean = input_tensor.mean(dim=tuple(range(1, input_tensor.dim())), keepdim=True)
    var = input_tensor.var(dim=tuple(range(1, input_tensor.dim())), unbiased=False, keepdim=True)

    normed_tensor = (input_tensor - mean) / torch.sqrt(var + eps)

    return normed_tensor

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# all_correct = True
# for dim_count in range(3, 9):
#     input_tensor = torch.randn(*list(range(3, dim_count + 2)), dtype=torch.float)
#     layer_norm = nn.LayerNorm(input_tensor.size()[1:], elementwise_affine=False, eps=eps)
#
#     norm_output = layer_norm(input_tensor)
#     custom_output = custom_layer_norm(input_tensor, eps)

#     all_correct &= torch.allclose(norm_output, custom_output, 1e-2)
#     all_correct &= norm_output.shape == custom_output.shape
# print(all_correct)
