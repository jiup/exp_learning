from sklearn.model_selection import KFold
import decision_tree_generator
from random import shuffle
from copy import copy
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from pt_mlp_classifier import *
import matplotlib.pyplot as plt

desc_nodes = decision_tree_generator.attr_nodes_from('data/iris.desc.discrete.txt')
data = decision_tree_generator.data_from(desc_nodes, 'data/iris.data.discrete.txt')
_query = [n for n in desc_nodes if n.value == 'Class'][0]
shuffle(data)

data_train_all = data[:100]
data_test_all = data[100:]

num_train = 0
correct_rate = []
num_train_list = []
kf = KFold(n_splits=5)
total_nn = 0
total_dtree = 0
X, result = load_data('data/iris.data.txt')
y = flatten(result, 3)
# X_ = torch.from_numpy(np.array(X).astype(np.float)).float()
# y_ = torch.from_numpy(np.array(y).astype(np.float)).float()
model = NNClassifier([4, 8, 8, 3])
#
for train_index, test_index in kf.split(X):
    train_data = torch.from_numpy(np.array([X[i] for i in train_index]).astype(np.float)).float()
    train_actual = torch.from_numpy(np.array([y[i] for i in train_index]).astype(np.float)).float()
    test_data = np.array([X[i] for i in test_index]).astype(np.float)
    test_actual = [np.array(y[i]) for i in test_index]
    model = training(model, train_data, train_actual, 0.007, 1000)
    pre = []
    # for i in test_data:
    #     result.append([evaluate(model, [i])])
    # result = flatten(result, 3)
    for i in test_data:
        result= evaluate(model, [i])
        pre.append(label_binarize([result],[0,1,2])[0])
    # print(pre)
    # print(test_actual)
    # a = test_actual.detach().numpy()
    # b = result.detach().numpy()
    # print(a, b)
    # print(a.shape, b.shape)
    # print(type(a), type(b))  # <class 'numpy.ndarray'>
    tmp_sum = 0
    for i in range(len(pre)):
        tmp_sum += average_precision_score(np.array(pre[i]),np.array(test_actual[i]))
    total_nn += tmp_sum/len(pre)

for train_index, test_index in kf.split(data):
    train_data = [data[i] for i in train_index]
    test_data = [data[i] for i in test_index]
    decision_tree = decision_tree_generator.generate_tree(_query, train_data, [n for n in desc_nodes if n != _query],
                                                          None)
    pre_data = []
    actual = []
    for i in test_data:
        copy(i).pop('Class')
        result = decision_tree_generator.evaluate(decision_tree, i).strip('?')
        pre_data.append((label_binarize([result], ['Iris-versicolor', 'Iris-virginica', 'Iris-setosa'])[0]))
        actual.append((label_binarize([i['Class']], ['Iris-versicolor', 'Iris-virginica', 'Iris-setosa'])[0]))
    a = np.array(pre_data)
    b = np.array(actual)
    # print(actual)#[array([1, 0, 0]), array([1, 0, 0]), array([1, 0, 0]), array([1, 0, 0])
    total_dtree += average_precision_score(np.array(actual), np.array(pre_data))
average_nn = total_nn / 5
average_dtre = total_dtree / 5
print(average_nn, average_dtre)
# decision_tree = generate_tree(_query, data_train_all, [n for n in desc_nodes if n != _query], None)
