import numpy as np
from enum import Enum
from collections import Counter


class DTNodeType(Enum):
    INTERNAL = 0
    LEAF = 1


class DTNode:
    def __init__(self, value, branches=None):
        self.value = value
        if branches is None:
            self.type = DTNodeType.LEAF
        else:
            self.type = DTNodeType.INTERNAL
            self.branches = branches

    def __str__(self):
        if self.type == DTNodeType.LEAF:
            return f"[{self.value}]"
        elif self.type == DTNodeType.INTERNAL:
            s = [f"({self.value})\n"]
            for k, v in self.branches.items():
                s.append("\t%s" % k)
                s.extend(map(lambda l: "\t%s\n" % l, str(v).splitlines()))
            return "".join(s)


def attr_nodes_from(file):
    nodes = []
    with open(file, 'r') as f:
        for line in f:
            attr, values = line.rstrip('\n').split(r': ')
            nodes.append(DTNode(attr, dict.fromkeys(values.split("/"))))
    return nodes


def data_from(nodes, file):
    attrs = list(map((lambda node: node.value), nodes))
    examples = []
    with open(file, 'r') as f:
        for line in f:
            examples.append(dict(zip(attrs, line.rstrip('\n').split(','))))
    return examples


def generate_tree(query, examples, attr_nodes, parent_examples):
    if not examples:
        return plurality_leaf(query.value, parent_examples)
    elif homo(query.value, examples):
        return DTNode(examples[0][query.value])
    elif not attr_nodes:
        return plurality_leaf(query.value, examples)
    else:
        attr_index = np.argmax(list(map(lambda n: importance(query, n, examples), attr_nodes)))
        # print(f"{attr_nodes[attr_index].value}\n")
        node = attr_nodes[attr_index]
        del attr_nodes[attr_index]
        for k in node.branches.keys():
            exs = extract_examples(node.value, k, examples)
            subtree = generate_tree(query, exs, attr_nodes, examples)
            node.branches[k] = subtree
        return node


def extract_examples(attr, value, examples):
    result = []
    for example in examples:
        if example[attr] == value:
            result.append(example)
    return result


def homo(query, examples):
    flag = None
    for example in examples:
        if flag is None:
            flag = example[query]
        elif example[query] != flag:
            return False
    return True


def plurality_leaf(query, examples):
    if examples is None:
        return DTNode('<Unknown>')
    return DTNode('P-' + Counter(list(map(lambda ex: ex[query], examples))).most_common(1)[0][0])


def importance(query, attribute, examples):
    attr = attribute.value
    length = len(examples)
    entropy = 0
    for k in query.branches.keys():
        k_length = sum(1 for example in examples if example[query.value] == k)
        tmp = k_length / length
        if tmp != 0:
            entropy += -tmp * np.log(tmp)

    v_entropies = []
    for v in attribute.branches.keys():
        v_length = sum(1 for example in examples if example[attr] == v)
        if v_length == 0:
            break
        v_entropy = 0
        for k in query.branches.keys():
            v_k_length = sum(1 for example in examples if example[attr] == v and example[query.value] == k)
            tmp2 = v_k_length / v_length
            if tmp2 != 0:
                v_entropy += -tmp2 * np.log(tmp2)
        v_entropies.append(v_entropy * v_length / length)

    # print(f"importance of {attr} is {entropy - sum(v_entropies)}")
    return entropy - sum(v_entropies)


def test():
    desc_nodes = attr_nodes_from('data/AIMA_Restaurant-desc.txt')
    data = data_from(desc_nodes, 'data/AIMA_Restaurant-data.txt')
    # print(*desc_nodes, sep='\n')
    # print(*data, sep='\n')
    _query = [n for n in desc_nodes if n.value == 'WillWait'][0]
    print(generate_tree(_query, data, [n for n in desc_nodes if n != _query], None))


def test2():
    desc_nodes = attr_nodes_from('data/iris.desc.discrete.txt')
    data = data_from(desc_nodes, 'data/iris.data.discrete.txt')
    # print(*desc_nodes, sep='\n')
    # print(*data, sep='\n')
    _query = [n for n in desc_nodes if n.value == 'Class'][0]
    print(generate_tree(_query, data, [n for n in desc_nodes if n != _query], None))


if __name__ == '__main__':
    test2()
