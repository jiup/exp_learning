from decision_tree_generator import *
from random import shuffle
import matplotlib.pyplot as plt


desc_nodes = attr_nodes_from('data/iris.desc.discrete.txt')
data = data_from(desc_nodes, 'data/iris.data.discrete.txt')
_query = [n for n in desc_nodes if n.value == 'Class'][0]
shuffle(data)

data_train_all = data[:100]
data_test_all = data[100:]

num_train = 0
correct_rate = []
num_train_list = []

for num_train in range(10, 100, 10):
    decision_tree = generate_tree(_query, data_train_all[:num_train], [n for n in desc_nodes if n != _query], None)
    num_correct = 0
    # print(decision_tree)
    for evid in data_test_all:
        klass = evaluate(decision_tree, evid)
        # print(klass, evid['Class'])
        if klass.strip('?') == evid['Class']:
            num_correct += 1
    correct_rate.append(num_correct/50)
    num_train_list.append(num_train)

def my_plotter(ax, data1, data2, param_dict):
    out = ax.plot(data1, data2, **param_dict)
    return out

fig, ax = plt.subplots(1, 1)
out = my_plotter(ax, num_train_list, correct_rate, {'marker': 'x'})

plt.xlabel('num_train')
plt.ylabel('cor_rate')
plt.title("Correct Rate as Training Set Grows")

plt.show()
