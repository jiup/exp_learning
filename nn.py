import torch
import torch.nn.functional as F

from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np


# torch.manual_seed(1)    # reproducible
def parse(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        return [list(map(lambda x: x.strip('\n'), line.split(','))) for line in lines]


a = parse("/Users/zhangyu/Ai/decision_tree/iris.data.txt")
# print(a)
x = np.array(a[:-1])
x1 = x[..., 0].astype(np.float64)
x2 = x[..., 1].astype(np.float64)
x3 = x[..., 2].astype(np.float64)
x4 = x[..., 3].astype(np.float64)
x5 = x[..., 4]
for i,d in enumerate(x5):
    if d == 'Iris-setosa':
        x5[i] = 1
    elif d == 'Iris-versicolor':
        x5[i] = 50
    else:
        x5[i] = 100
input = torch.from_numpy(np.c_[x1,x2,x3,x4].astype(np.float)).float()
output = torch.from_numpy(np.c_[x5].astype(np.float)).float()
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


net = Net(n_feature=4, n_hidden=5, n_output=1)  # define the network


optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

# torch_dataset = Data.TensorDataset(data_tensor=input, target_tensor=output)

for t in range(20000):
    prediction = net(input)  # input x and predict based on x

    loss = loss_func(prediction, output)  # must be (1. nn output, 2. target)
    print(loss)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
pre = net(input)
print(pre)