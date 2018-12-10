import numpy as np
import copy
import math
import random


# with open("/Users/zhangyu/Ai/decision_tree/AIMA_Restaurant-desc.txt") as f:
#     lines = f.readlines()
#     attr_names = [line.split(" ") for line in lines]
#
# print(attr_names)

def parse(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        return [list(map(lambda x: x.strip('\n'), line.split(','))) for line in lines]


class Dtree:
    def __init__(self, name, value=None):
        self.attrs = name
        self.value = value
        self.children = {}


class Dtree_learner:
    def __init__(self, file_name, info_file_name=None, ):
        self.datalist = parse(file_name)  # a list to store db content. each line is a row
        self.attrs_table = {}

        if info_file_name is not None:
            with open(info_file_name, "r") as f:
                lines = f.readlines()
                self.attrs = [line.split(" ")[0] for line in lines]
        else:
            self.attrs = [i for i in range(len(self.datalist[0]))]
        # print(self.attrs)
        self.label = self.attrs[-1]
        for idx, attr in enumerate(self.attrs):
            self.attrs_table[attr] = idx

    def decision_tree_learning(self, examples, attrs, parent_examples=tuple()):
        # print(attrs)
        if len(examples) == 0:
            return self.plurality_value(parent_examples)
        elif self.same_class(examples):
            return Dtree('result', examples[0][-1])
        elif len(attrs) == 1:
            return self.plurality_value(examples)
        else:
            attr = self.get_attr(attrs, examples)
            if attr is None:
                print('d')
            node = Dtree(attr)
            val_map = self.seperate_by_value(attr, examples)
            for key, value in val_map.items():
                attr_copy = copy.copy(attrs)
                attr_copy.remove(attr)
                # print(key)
                node.children[key] = self.decision_tree_learning(value, attr_copy, examples)
        return node

    def seperate_by_value(self, attr, examples):
        seperated_table = {}
        # print(attr)
        idx = self.attrs_table[attr]
        for example in examples:
            if example[idx] in seperated_table.keys():
                seperated_table[example[idx]].append(example)
            else:
                seperated_table[example[idx]] = [example]
        return seperated_table

    def get_attr(self, attrs, examples):
        total_info = self.get_infor(examples)
        max_info = -float("inf")
        max_attr = None
        for attr in attrs:
            if attr == self.label:
                continue
            val_map = self.seperate_by_value(attr, examples)
            after_info = 0
            for k, v in val_map.items():
                cnt = len(v)
                attr_info = self.get_infor(v)
                after_info += cnt * attr_info
            if total_info - after_info > max_info:
                max_info = total_info - after_info
                max_attr = attr
        return max_attr

    def get_infor(self, examples, attrs=None):
        if attrs is None:
            attrs = self.label
        count_map = {}
        val_map = self.seperate_by_value(attrs, examples)
        total_count = len(examples)
        infor = 0
        for i in val_map.keys():
            p = len(val_map[i]) / total_count
            infor += (-p) * math.log2(p)
        return infor

    def plurality_value(self, examples):
        label_map = self.seperate_by_value(self.label, examples)
        p = random.uniform(0, 1)
        total_cnt = len(examples)
        min_dis = 1
        min_val = None
        for val in label_map.values():
            dis = abs(p - (len(val) / total_cnt))
            if dis < min_dis:
                min_dis = dis
                min_val = val[-1]
        return Dtree("result", min_val)

    def same_class(self, examples):
        tmp = examples[0][-1]
        for example in examples:
            if example[-1] != tmp:
                return False
        return True

    def predict(self, example):
        # TODO
        pass


def dfs(root):
    if root.attrs == "result":
        print(root.value)
        return
    print(root.attrs)
    for child in root.children.values():
        dfs(child)


if __name__ == '__main__':
    dl = Dtree_learner("/Users/zhangyu/Ai/decision_tree/iris.data.discrete.txt")
    tree = dl.decision_tree_learning(dl.datalist, dl.attrs)
    dfs(tree)
