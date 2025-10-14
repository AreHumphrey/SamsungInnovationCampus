import torch

w = torch.tensor([[5.0, 10.0], [1.0, 2.0]], requires_grad=True)

function = torch.prod(torch.log(torch.log(w + 7)))
function.backward()