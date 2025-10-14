import torch

N = 4
C = 3
C_out = 10
H = 8
W = 16

x = torch.ones((N, C, H, W))
out1 = torch.nn.Conv2d(C, C_out, kernel_size=(3, 3), padding=1)(x)
out2 = torch.nn.Conv2d(C, C_out, kernel_size=(5, 5), padding=2)(x)
out3 = torch.nn.Conv2d(C, C_out, kernel_size=(7, 7), padding=3)(x)
out4 = torch.nn.Conv2d(C, C_out, kernel_size=(9, 9), padding=4)(x)
out5 = torch.nn.Conv2d(C, C_out, kernel_size=(3, 5), padding=(1, 2))(x)
out6 = torch.nn.Conv2d(C, C_out, kernel_size=(3, 3), padding=8)(x)
out7 = torch.nn.Conv2d(C, C_out, kernel_size=(4, 4), padding=1)(x)
out8 = torch.nn.Conv2d(C, C_out, kernel_size=(2, 2), padding=1)(x)