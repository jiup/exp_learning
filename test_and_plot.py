from decision_tree_generator import *
from random import shuffle
import matplotlib.pyplot as plt


def my_plotter(ax, data1, data2, param_dict):
    out = ax.plot(data1, data2, **param_dict)
    return out

# desc_file = 'data/iris.desc.discrete.txt'
# disc_data = 'data/iris.data.discrete.txt'
# num_example = 150
# klass_attr = "Class"
def test_and_plot(desc_file, disc_data, num_example, klass_attr):
    desc_nodes = attr_nodes_from(desc_file)
    data = data_from(desc_nodes, disc_data)
    _query = [n for n in desc_nodes if n.value == klass_attr][0]
    shuffle(data)

    # data_train_all = data[:100]
    # data_test_all = data[100:]

    correct_rate = []
    num_train_list = []

    for num_train in range(1, num_example, 1):
        data_train = data[:num_train]
        decision_tree = generate_tree(_query, data_train, [n for n in desc_nodes if n != _query], None)
        data_test_all = data[num_train:]
        num_correct = 0
        # print(decision_tree)

        for r in range(100):
            total_correct_rate = 0.0
            for evid in data_test_all:
                klass = evaluate(decision_tree, evid)
                # print(klass, evid['Class'])
                if klass.strip('?') == evid[klass_attr]:
                    num_correct += 1
            total_correct_rate += num_correct/(num_example-num_train)

        correct_rate.append(total_correct_rate/100)
        num_train_list.append(num_train/num_example)
        # shuffle(data)



    fig, ax = plt.subplots(1, 1)
    out = my_plotter(ax, num_train_list, correct_rate, {'marker': '.'})

    plt.xlabel('training_size')
    plt.ylabel('correct_rate')
    plt.title("Correct Rate as Training Set Grows")

    plt.show()

# test iris
# test_and_plot('data/iris.desc.discrete.txt', 'data/iris.data.discrete.txt', 150, "Class")

test_and_plot('data/AIMA_Restaurant-desc.txt', 'data/AIMA_Restaurant-data.txt', 12, 'WillWait')

