from matplotlib import pyplot as plt
import numpy as np
import torch
from pt_mlp_classifier import *

X, result = load_data('data/iris.data.txt')
# print(result)
y = flatten(result, 3)
# print(y)
X = torch.from_numpy(np.array(X).astype(np.float)).float()
y = torch.from_numpy(np.array(y).astype(np.float)).float()
model = NNClassifier([4, 8, 8, 3])
model2 = NNClassifier([4, 8, 8, 3])
model3 = NNClassifier([4, 8, 8, 3])
# print(X.shape,y[:int(0.3*len(X))].shape)

# model_1, loss_1 = training(model, X[:int(0.3*len(X))], y[:int(0.3*len(X))], 0.007, 200, True)
# model_3, loss_3 = training(model2, X[:int(0.6*len(X))], y[:int(0.6*len(X))], 0.007, 200, True)
# model_2, loss_2 = training(model3, X[:int(0.9*len(X))], y[:int(0.9*len(X))], 0.007, 200, True)
model_1, loss_1 = training(model, X, y, 0.9, 5000, True)
model_2, loss_2 = training(model2, X, y, 0.09, 5000, True)
model_3, loss_3 = training(model3, X, y, 0.009, 5000, True)


# print(len(loss_2))
x_= [i for i in range(0,5000,10)]
# print(len(x_))
plt.plot(x_, loss_1[::10], color='green', label='learning rate = 0.9')
plt.plot(x_, loss_2[::10], color='red', label='learning rate = 0.09')
plt.plot(x_, loss_3[::10], color='skyblue', label='learning rate = 0.009')
# plt.ylim(0,0.1)
plt.title('loss vs times of training')
# plt.plot(x, y_rejection, color='blue', label='my rejection')
plt.legend()  # 显示图例
plt.xlabel('num training')
plt.ylabel('loss')
plt.show()
