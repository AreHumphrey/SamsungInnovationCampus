import torch
import torch.nn as nn

channel_count = 6
eps = 1e-3
batch_size = 20
input_size = 2

input_tensor = torch.randn(batch_size, channel_count, input_size)


def custom_group_norm(input_tensor, num_groups, eps):
    batch_size, channels, length = input_tensor.shape

    assert channels % num_groups == 0, "Number of channels must be divisible by num_groups"

    channels_per_group = channels // num_groups
    grouped = input_tensor.view(batch_size, num_groups, channels_per_group, length)

    mean = grouped.mean(dim=(2, 3), keepdim=True)
    var = grouped.var(dim=(2, 3), unbiased=False, keepdim=True)

    normed_grouped = (grouped - mean) / torch.sqrt(var + eps)

    normed_tensor = normed_grouped.view(batch_size, channels, length)

    return normed_tensor
