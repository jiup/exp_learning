import torch
import torch.nn.functional as F
import numpy as np

torch.manual_seed(1)  # reproducible


def parse(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        return [list(map(lambda x: x.strip('\n'), line.split(','))) for line in lines]


a = parse("/Users/zhangyu/Ai/decision_tree/iris.data.txt")
x = np.array(a[:-1])
x1 = x[..., 0].astype(np.float64)
x2 = x[..., 1].astype(np.float64)
x3 = x[..., 2].astype(np.float64)
x4 = x[..., 3].astype(np.float64)
x5 = x[..., 4]
for i, d in enumerate(x5):
    if d == 'Iris-setosa':
        x5[i] = 1
    elif d == 'Iris-versicolor':
        x5[i] = 2
    else:
        x5[i] = 3
x6 = []
for i in x5:
    if i == '1':
        x6.append([1, 0, 0])
    elif i == '2':
        x6.append([0, 1, 0])
    else:
        x6.append([0, 0, 1])
input = torch.from_numpy(np.c_[x1, x2, x3, x4].astype(np.float)).float()
output = torch.from_numpy(np.c_[x6].astype(np.float)).float()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        return F.softmax(self.predict(x), dim=1)


net = Net(n_feature=4, n_hidden=8, n_output=3)  # define the network

optimizer = torch.optim.RMSprop(net.parameters(), lr=0.0015)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
for t in range(1000):
    prediction = net(input)  # input x and predict based on x
    loss = loss_func(prediction, output)  # must be (1. nn output, 2. target)
    print(loss)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
pre = net(input)
print(pre)
