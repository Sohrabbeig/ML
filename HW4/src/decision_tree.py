def split(index, value, dataset):
    left, right = list(), list()
    for item in dataset:
        if item[index] < value:
            left.append(item)
        else:
            right.append(item)
    return left, right


def information_gain(impurity_measure, dataset, index, item):
    class_values = list(set(item[-1] for item in dataset))
    left, right = split(index, item[index], dataset)
    left_count = len(left)
    right_count = len(right)
    total_count = left_count + right_count
    if left_count == 0 or right_count == 0:
        return 0
    return (impurity_measure(dataset, class_values) \
            - (left_count / total_count) * impurity_measure(left, class_values) \
            - (right_count / total_count) * impurity_measure(right, class_values))


def get_split(dataset, impurity_measure):
    b_index, b_value, b_score, b_groups = float('-inf'), float('-inf'), float('-inf'), None
    for index in range(len(dataset[0]) - 1):
        for item in dataset:
            ig = information_gain(impurity_measure, dataset, index, item)
            if ig > b_score:
                b_index, b_value, b_score, b_groups = index, item[index], ig, split(index, item[index], dataset)
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


class Node:
    def __init__(self, index = None, value = None, left = None, right = None):
        self.index = index
        self.value = value
        self.left = left
        self.right = right
        self.prediction = None


class Tree:
    def __init__(self, impurity_measure, dataset, max_depth=float('inf'), min_size=1):
        self.root = None
        self.impurity_measure = impurity_measure
        self.dataset = dataset
        self.max_depth = max_depth
        self.min_size = min_size

    def build(self):

        def is_pure(dataset):
            class_values = set()
            for item in dataset:
                class_values.add(item[-1])
            return len(class_values) == 1 or len(class_values) == 0

        def process(dataset, depth):
            if not is_pure(dataset) and len(dataset) > self.min_size and depth < self.max_depth:
                best_split = get_split(dataset, self.impurity_measure)
                current_node = Node(index=best_split['index'],
                                    value=best_split['value'],
                                    left=process(best_split['groups'][0], depth+1),
                                    right=process(best_split['groups'][1], depth+1)
                                    )
            else:
                current_node = Node()
                outcomes = [row[-1] for row in dataset]
                print(outcomes)
                current_node.prediction = max(set(outcomes), key=outcomes.count)

            return current_node

        self.root = process(self.dataset, depth=0)

    def predict(self, item, current_node=None):
        if current_node == None:
            current_node = self.root
        if not current_node.prediction == None:
            return current_node.prediction
        elif item[current_node.index] < current_node.value:
            return self.predict(item, current_node.left)
        else:
            return self.predict(item, current_node.right)

    def accuracy(self, test_dataset):
        true_predictions = 0
        for test in test_dataset:
            if self.predict(test[:-1]) == test[-1]:
                true_predictions += 1
        return true_predictions / len(test_dataset)