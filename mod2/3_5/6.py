import torch


def target_function(x):
    return 2 ** x * torch.sin(2 ** -x)


class RegressionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 50)
        self.act = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(50, 1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


net = RegressionNet()

x_train = torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)


optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


def loss(pred, target):
    return ((pred - target) ** 2).mean()


for epoch_index in range(3000):
    optimizer.zero_grad()
    y_pred = net(x_train)
    loss_value = loss(y_pred, y_train)
    loss_value.backward()
    optimizer.step()