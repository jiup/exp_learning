import numpy as np
import random


class BPNN:
    def __init__(self, sizes, act_func, cost_func, softmax=True):
        self.l_size = len(sizes)
        self.sizes = sizes
        self.biases = np.array([np.random.randn(i, 1) for i in sizes[1:]])
        self.weights = np.array([np.random.randn(k, j) for j, k in zip(sizes[:-1], sizes[1:])])
        self.activation, self.activation_d = act_func
        self.cost, self.cost_d = cost_func
        self.softmax_on = softmax

    def sgd(self, xs, ys, alpha, max_iter):
        iter_count = 0
        while iter_count < max_iter:
            acc_dbs = acc_dws = None
            training_pairs = list(zip(xs, ys))
            random.shuffle(training_pairs)
            for x, y in training_pairs:
                dbs, dws = self.backprop(x, y)
                acc_dbs, acc_dws = [dbs, dws] if acc_dbs is None else [np.add(acc_dbs, dbs), np.add(acc_dws, dws)]
                self.biases = [b - alpha * db for b, db in zip(self.biases, dbs)]
                self.weights = [w - alpha * dw for w, dw in zip(self.weights, dws)]
                iter_count = iter_count + 1
            # self.biases = np.subtract(self.biases, alpha * acc_dbs)
            # self.weights = np.subtract(self.weights, alpha * acc_dws)
            # # self.biases = [b - alpha * acc_dbs for b, db in zip(self.biases, acc_dbs)]
            # # self.weights = [w - alpha * acc_dws for w, dw in zip(self.weights, acc_dws)]
            # iter_count = iter_count + 1
            # show loss improvement
            h = self.evaluate(np.vstack([5.9, 3.0, 5.1, 1.8]))
            if iter_count % 1000 == 0:
                print(f"iter_count = {iter_count}")
                print(h, self.cost(h, [0, 0, 1]))
        pass

    def evaluate(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.dot(w, a) + b)
        return self.softmax(a)

    def feed_forward(self, a):
        zs = []
        acts = []
        for w, b in zip(self.weights, self.biases):
            acts.append(a)
            zs.append(np.dot(w, a) + b)
            a = self.activation(zs[-1])
        acts.append(self.softmax(zs[-1]) if self.softmax_on else a)
        return acts, zs

    def backprop(self, x, y):
        dws = [np.zeros(np.shape(w)) for w in self.weights]
        dbs = [np.zeros(np.shape(b)) for b in self.biases]
        acts, zs = self.feed_forward(x)
        final_activation_d = self.softmax_d if self.softmax_on else self.activation_d
        # ∂C/∂b = ∂C/∂a * ∂a/∂z * ∂z/∂b(=1) = ∂C/∂a * ∂a/∂z <==> error
        dbs[-1] = self.cost_d(acts[-1], y) * final_activation_d(zs[-1])
        # ∂C/∂w = ∂C/∂a * ∂a/∂z * ∂z/∂w(=prev_a) = ∂C/∂b * prev_a
        dws[-1] = np.dot(dbs[-1], acts[-2].transpose())
        for layer in range(-2, -self.l_size, -1):
            dbs[layer] = np.dot(self.weights[layer + 1].transpose(), dbs[layer + 1]) * self.activation_d(zs[layer])
            dws[layer] = np.dot(dbs[layer], np.transpose(acts[layer - 1]))
        return dbs, dws

    def softmax(self, ys):
        # print(f"\nhs = \n{ys}")
        y_exp = np.exp(ys - np.max(ys))
        # print(f"normalized (max={max(ys)}) = \n{y_exp}")
        # print(f"sum = {sum(y_exp)}")
        # print(f"softmax = {y_exp / sum(y_exp)}")
        return y_exp / sum(y_exp)

    def softmax_d(self, ys):
        # return ys * (1 - ys)
        return np.vstack(np.sum(np.diagflat(ys) - np.dot(ys, ys.T), axis=1))


# Activation functions
def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    return np.array([1 / (1 + np.exp(-i)) if i >= 0 else np.exp(i) / (1 + np.exp(i)) for i in x])


def delta_sigmoid(x):
    tmp = sigmoid(x)
    return tmp * (1 - tmp)


def relu(x, leaky=0):
    return np.maximum(0, x) if leaky == 0 else np.maximum(leaky * x, x)


def delta_relu(x, leaky=0):
    return np.where(x <= 0, 0, 1) if leaky == 0 else np.where(x <= 0, leaky, 1)


# Loss functions
def mean_squared_loss(hs, ys):
    return np.mean([np.square(yi - hi) for hi, yi in zip(hs, ys)])


def delta_mean_squared_loss(hs, ys):
    return 2 * np.subtract(hs, ys)


def cross_entropy_loss(hs, ys, fix=1e-12):
    hs = np.clip(hs, fix, 1 - fix)  # fix to avoid log(0)
    return -np.sum(ys * np.log(hs)) / len(hs)


def delta_cross_entropy_loss(hs, ys, fix=1e-12):
    hs = np.clip(hs, fix, 1 - fix)  # fix to avoid log(0)
    # ∂C/∂hi = – yi / hi + (1 – yi)/ (1 – hi)
    return np.where(ys == 1, -1 / hs, 1 / (1 - hs))


def test():
    classifier = BPNN([2, 3, 4], [relu, delta_relu], [cross_entropy_loss, delta_cross_entropy_loss])
    # print(network.softmax([1, 2, 3]))
    # print(ce_cost(np.array([[0.25,0.25,0.25,0.25],
    #                         [0.01,0.01,0.01,0.96]]),
    #               np.array([[0,0,0,1],
    #                    [0,0,0,1]])))
    # print(ce_cost([1, 0, 0, 1], [0, 1, 1, 0]))
    # print(ce_cost([0, 1, 1, 0], [0, 1, 1, 0]))
    # print(network.feed_forward(np.vstack([1, 2])))
    classifier.sgd(
        [[[0], [1]], [[1], [0]]],
        [[[1], [0], [0], [0]], [[0], [1], [0], [0]]],
        0.0003, 100)


def test_iris_classifier():
    # fill in the iris dataset
    xs, ys, y_dict = [], [], {}
    with open('data/iris.data.txt', 'r') as f:
        for line in f:
            cells = line.rstrip('\n').split(',')
            out = cells[-1]
            if out not in y_dict:
                y_dict[out] = len(y_dict)
            xs.append(np.vstack([float(i) for i in cells[:-1]]))
            ys.append(y_dict[out])
    print(y_dict)

    iris_classifier = BPNN(
        [4, 4, 3],
        [sigmoid, delta_sigmoid],
        [cross_entropy_loss, delta_cross_entropy_loss])

    # print(iris_classifier.softmax(np.vstack([1, 2.5, 3, 0])))
    iris_classifier.sgd(xs, ys, 1e-13, 500000)


if __name__ == '__main__':
    np.seterr(all='warn', over='raise')
    test_iris_classifier()
