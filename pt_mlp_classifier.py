import torch
import numpy as np

torch.manual_seed(1)


class NNClassifier(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(NNClassifier, self).__init__()
        self.hidden_layer = torch.nn.Linear(n_input, n_hidden)
        self.output_layer = torch.nn.Linear(n_hidden, n_output)

    def forward(self, a):
        a = torch.nn.functional.relu(self.hidden_layer(a))
        return torch.nn.functional.softmax(self.output_layer(a), dim=1)


def load_data(path):
    with open(path, 'r') as f:
        data = list(line.strip('\n').split(',') for line in f.readlines())
    data = np.array(data)
    X = data[:, range(0, data.shape[1] - 1)].astype(np.float)
    y = data[:, [data.shape[1] - 1]]
    y_dict = {}
    for cell in y:
        if cell[0] not in y_dict:
            y_dict[cell[0]] = len(y_dict)
        cell[0] = y_dict[cell[0]]
    return X, y


def flatten(y, n):
    new_y = []
    for i, y_idx in enumerate(y):
        new_y.append([1 if i == int(y_idx[0]) else 0 for i in range(n)])
    return new_y


def test():
    X, y = load_data('data/iris.data.txt')
    y = flatten(y, 3)
    X = torch.from_numpy(np.array(X).astype(np.float)).float()
    y = torch.from_numpy(np.array(y).astype(np.float)).float()
    # print(X, '\n', y)

    classifier = NNClassifier(4, 8, 3)
    optimizer = torch.optim.RMSprop(classifier.parameters(), lr=0.0015)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    for it in range(100):
        prediction = classifier.forward(X)  # input x and predict based on x
        loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)
        print(loss)
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients


if __name__ == '__main__':
    test()
