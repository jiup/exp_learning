import numpy as np


class BPNN:
    def __init__(self, sizes, act_func, cost_func, softmax=True):
        self.l_size = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]
        self.weights = [np.random.randn(k, j) for j, k in zip(sizes[:-1], sizes[1:])]
        self.activation, self.activation_d = act_func
        self.cost, self.cost_d = cost_func
        self.softmax_on = softmax

    def sgd(self, xs, ys, alpha):
        for x, y in zip(xs, ys):
            dbs, dws = self.backprop(x, y)
            self.biases = [b - alpha * db for b, db in zip(self.biases, dbs)]
            self.weights = [w - alpha * dw for w, dw in zip(self.weights, dws)]
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
        print(acts[-1], y)
        dbs[-1] = self.cost_d(acts[-1], y) * final_activation_d(zs[-1])
        # ∂C/∂w = ∂C/∂a * ∂a/∂z * ∂z/∂w(=prev_a) = ∂C/∂b * prev_a
        dws[-1] = np.dot(dbs[-1], acts[-2].transpose())
        for layer in range(-2, -self.l_size, -1):
            dbs[layer] = np.dot(self.weights[layer + 1].transpose(), dbs[layer + 1]) * self.activation_d(zs[layer])
            dws[layer] = np.dot(dbs[layer], np.transpose(acts[layer - 1]))
        return dbs, dws

    def softmax(self, ys):
        y_exp = np.exp(ys - np.max(ys))
        return y_exp / sum(y_exp)

    def softmax_d(self, ys):
        return ys * (1 - ys)


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def delta_sigmoid(x):
    return x * (1 - x)


def relu(x, leaky=0):
    return np.maximum(leaky * x, x)


def delta_relu(x, leaky=0):
    return np.where(x <= 0, leaky, 1)


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
        0.0003
    )


if __name__ == '__main__':
    test()
