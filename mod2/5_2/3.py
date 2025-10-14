import torch
from abc import ABC, abstractmethod


def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    batch_size, channels_count, input_height, input_width = input_matrix_shape
    output_height = (input_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    return batch_size, out_channels, output_height, output_width


class ABCConv2d(ABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def set_kernel(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, input_tensor):
        pass


class Conv2d(ABCConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding=0, bias=False)

    def set_kernel(self, kernel):
        self.conv2d.weight.data = kernel

    def __call__(self, input_tensor):
        return self.conv2d(input_tensor)


def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
    out_channels = kernel.shape[0]
    in_channels = kernel.shape[1]
    kernel_size = kernel.shape[2]

    layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
    layer.set_kernel(kernel)

    return layer(input_matrix)


def test_conv2d_layer(conv2d_layer_class, batch_size=2,
                      input_height=4, input_width=4, stride=2):
    kernel = torch.tensor(
        [[[[0., 1, 0],
           [1, 2, 1],
           [0, 1, 0]],

          [[1, 2, 1],
           [0, 3, 3],
           [0, 1, 10]],

          [[10, 11, 12],
           [13, 14, 15],
           [16, 17, 18]]]])

    in_channels = kernel.shape[1]

    input_tensor = torch.arange(0, batch_size * in_channels *
                                input_height * input_width,
                                out=torch.FloatTensor()) \
        .reshape(batch_size, in_channels, input_height, input_width)

    custom_conv2d_out = create_and_call_conv2d_layer(
        conv2d_layer_class, stride, kernel, input_tensor)
    conv2d_out = create_and_call_conv2d_layer(
        Conv2d, stride, kernel, input_tensor)

    return torch.allclose(custom_conv2d_out, conv2d_out) \
        and (custom_conv2d_out.shape == conv2d_out.shape)


class Conv2dLoop(ABCConv2d):
    def __call__(self, input_tensor):
        batch_size, in_channels, input_height, input_width = input_tensor.shape
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        stride = self.stride

        _, _, out_height, out_width = calc_out_shape(
            (batch_size, in_channels, input_height, input_width),
            out_channels,
            kernel_size,
            stride,
            0
        )

        output_tensor = torch.zeros(batch_size, out_channels, out_height, out_width, dtype=input_tensor.dtype)


        for b in range(batch_size):

            for oc in range(out_channels):

                for oh in range(out_height):

                    for ow in range(out_width):

                        for ic in range(in_channels):

                            for kh in range(kernel_size):

                                for kw in range(kernel_size):

                                    ih = oh * stride + kh
                                    iw = ow * stride + kw

                                    output_tensor[b, oc, oh, ow] += (
                                            input_tensor[b, ic, ih, iw] *
                                            self.kernel[oc, ic, kh, kw]
                                    )

        return output_tensor