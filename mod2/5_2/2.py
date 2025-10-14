import numpy as np


def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    batch_size, in_channels, height, width = input_matrix_shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    return [batch_size, out_channels, out_height, out_width]

print(np.array_equal(
    calc_out_shape(input_matrix_shape=[2, 3, 10, 10],
                   out_channels=10,
                   kernel_size=3,
                   stride=1,
                   padding=0),
    [2, 10, 8, 8]))

