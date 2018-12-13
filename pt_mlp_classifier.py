import torch
import numpy as np
import sys

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
    optimizer = torch.optim.RMSprop(classifier.parameters(), lr = alpha)
    loss_func = torch.nn.MSELoss()
    loss_history = []
    for epoch in range(iter_count):
        for i in range(0,len(X),30):
            prediction = classifier.forward(X[i:i+1])
            loss = loss_func(prediction, y[i:i+1])
            if i/10 == 0:
                print("loss: ",loss)
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


if __name__ == '__main__':
    torch.manual_seed(1)  # reproducible switch
    if sys.argv[1] == 'data/tic-tac-toe.data.txt':
        if sys.argv[2] == "2":
            model = torch.load('ttt_model.pkl')
            print("example:[1, 2, 2, 0, 1, 0, 0, 0, 1])")
            print(f"Result: class_{evaluate(model, [[1, 2, 2, 0, 1, 0, 0, 0, 1]])}")
            print("example:[0, 0, 1, 0, 1, 1, 2, 2, 2])")
            print(f"Result: class_{evaluate(model, [[0, 0, 1, 0, 1, 1, 2, 2, 2]])}")
            sys.exit()
        X, result = load_data(sys.argv[1])
        y = flatten(result, 2)
        X = torch.from_numpy(np.array(X).astype(np.float)).float()
        y = torch.from_numpy(np.array(y).astype(np.float)).float()
        # # print(X, '\n', y)
        model = NNClassifier([9, 81, 9, 2])
        model = training(model, X, y, 0.007, 500)
        torch.save(model, 'ttt_model.pkl')

    elif sys.argv[1] == 'data/iris.data.txt':
        if sys.argv[2] == "2":
            model = torch.load('iris_model.pkl')
            print("example:[5.1,3.8,1.4,0.2])")
            print(f"Result: class_{evaluate(model, [[5.1,3.8,1.4,0.2]])}")
            print("example:[6.0,3.0,4.6,1.8])")
            print(f"Result: class_{evaluate(model, [[6.0,3.0,4.6,1.8]])}")
            sys.exit()
        # print(sys.argv[1])
        X, result = load_data(sys.argv[1])
        y = flatten(result, 3)
        X = torch.from_numpy(np.array(X).astype(np.float)).float()
        y = torch.from_numpy(np.array(y).astype(np.float)).float()
        # # print(X, '\n', y)
        model = NNClassifier([4, 8, 3])
        model = training(model, X, y, 0.01, 500)
        torch.save(model, 'iris_model.pkl')

