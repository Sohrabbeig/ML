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
        i = 1
        for item in dataset:
            ig = information_gain(impurity_measure, dataset, index, item)
            print("index: {}, item: {}, ig: {}".format(index, i, ig))
            i += 1
            if ig > b_score:
                b_index, b_value, b_score, b_groups = index, item[index], ig, split(index, item[index], dataset)
            print(b_score)
    return {'index': b_index, 'value': b_value, 'groups': b_groups}
