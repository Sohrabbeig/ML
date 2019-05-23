from math import log2


def class_probabilities(data, classes):
    class_count = {}
    class_probability = {}

    for c in classes:
        class_count[c] = 0

    for item in data:
        class_count[item[-1]] += 1

    for c, n in class_count.items():
        class_probability[c] = n / len(data)

    return class_probability.items()


def gini_index(data, classes):
    return 1 - sum(p[1] ** 2 for p in class_probabilities(data, classes))


def entropy(data, classes):
    try:
        return -sum(p[1] * log2(p[1]) for p in class_probabilities(data, classes))
    except:
        return 0.0


def miss_classification(data, classes):
    return 1 - max(class_probabilities(data, classes), key=lambda item: item[1])[1]

#---------------testing impurity measures---------------
# split1 = [[1, 2, 3, 1],
#           [1, 3, 2, 1],
#           [1, 3, 2, 1],
#           [2, 3, 1, 0],
#           [2, 3, 1, 0],
#           [2, 1, 3, 0]]
#
# split2 = [[1, 2, 3, 1],
#           [1, 3, 2, 1],
#           [2, 3, 1, 1],
#           [2, 1, 3, 1]]
#
# classes = {0, 1}
#
# print(miss_classification(split1, classes))
# print(entropy(split1, classes))
# print(gini_index(split1, classes))
#
# print(miss_classification(split2, classes))
# print(entropy(split2, classes))
# print(gini_index(split2, classes))
