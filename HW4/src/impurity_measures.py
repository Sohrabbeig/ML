from math import log2


def class_probabilities(dataset, class_values):
    class_count = {}
    class_probability = {}
    for c in class_values:
        class_count[c] = 0

    for item in dataset:
        class_count[item[-1]] += 1

    for c, n in class_count.items():
        class_probability[c] = n / len(dataset)

    return class_probability.items()


def gini_index(dataset, class_values):
    return 1 - sum(p[1] ** 2 for p in class_probabilities(dataset, class_values))


def entropy(dataset, class_values):
    try:
        return -sum(p[1] * log2(p[1]) for p in class_probabilities(dataset, class_values))
    except:
        return 0.0


def miss_classification(dataset, class_values):
    return 1 - max(class_probabilities(dataset, class_values), key=lambda item: item[1])[1]

# ---------------testing impurity measures---------------
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
# class_values = [0, 1]
#
# print(miss_classification(split1, class_values))
# print(entropy(split1, class_values))
# print(gini_index(split1, class_values))
# 
# print(miss_classification(split2, class_values))
# print(entropy(split2, class_values))
# print(gini_index(split2, class_values))
