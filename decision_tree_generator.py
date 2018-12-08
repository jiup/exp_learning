from numpy import *
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
        return plurality_leaf(query, parent_examples)
    elif homo(query, examples):
        return DTNode(examples[0][query])
    elif not attr_nodes:
        return plurality_leaf(query, examples)
    else:
        attr_index = argmax(list(map(lambda n: importance(n.value, examples), attr_nodes)))
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


def importance(attribute, examples):
    # TODO
    if attribute == 'Patrons':
        return 5
    if attribute == 'Hungry':
        return 4
    if attribute == 'Type':
        return 3
    if attribute == 'Fri/Sat':
        return 2
    return 0


desc_nodes = attr_nodes_from('AIMA_Restaurant-desc.txt')
data = data_from(desc_nodes, 'AIMA_Restaurant-data.txt')
# print(*desc_nodes, sep='\n')
# print(*data, sep='\n')
_query = 'WillWait'
print(generate_tree(_query, data, [n for n in desc_nodes if n.value != _query], None))
