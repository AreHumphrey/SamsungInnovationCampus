import torch

x = torch.tensor([[10., 20.]])

fc = torch.nn.Linear(2, 3)

fc.weight.data = torch.tensor([[11., 12.],
                               [21., 22.],
                               [31., 32.]], dtype=torch.float32)

fc.bias.data = torch.tensor([31., 32., 33.], dtype=torch.float32)

fc_out = fc(x)

fc_out_alternative = x @ fc.weight.T + fc.bias
