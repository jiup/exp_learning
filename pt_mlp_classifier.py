import torch
import numpy as np


class NNClassifier(torch.nn.Module):
    def __init__(self, sizes):
        super(NNClassifier, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, a):
        for layer in self.layers[:-1]:
            a = torch.nn.functional.relu(layer(a))
        return torch.nn.functional.softmax(self.layers[-1](a), dim=1)


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


def training(classifier, X, y, alpha, iter_count, record_loss=False):
    optimizer = torch.optim.SGD(classifier.parameters(), alpha)
    loss_func = torch.nn.MSELoss()
    loss_history = []
    for epoch in range(iter_count):
        prediction = classifier.forward(X)
        loss = loss_func(prediction, y)
        if record_loss:
            loss_history.append(float(loss.data))
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        # print(f"Expected: {list(clazz.T.astype(np.int)[0])}")
        # print(f"Actual:   {[np.argmax(result) for result in prediction.data.numpy().squeeze()]}")
    return (classifier, loss_history) if record_loss else classifier


def evaluate(nn, x):
    x = torch.from_numpy(np.array(x).astype(np.float)).float()
    prediction = nn.forward(x)
    return np.argmax(prediction.data.numpy())
    # return prediction.data.numpy()


if __name__ == '__main__':
    torch.manual_seed(1)  # reproducible switch
    X, result = load_data('data/iris.data.txt')
    # print(result)
    y = flatten(result, 3)
    # print(y)
    X = torch.from_numpy(np.array(X).astype(np.float)).float()
    y = torch.from_numpy(np.array(y).astype(np.float)).float()
    model = NNClassifier([4, 8, 8, 3])
    model = training(model, X, y, 0.007, 400)
    print(f"Result: class_{evaluate(model, [[5.1000, 3.5000, 1.4000, 0.2000]])}")
