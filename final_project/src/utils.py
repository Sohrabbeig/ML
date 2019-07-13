from math import log2
import pandas as pd
import numpy as np
from sklearn import preprocessing


# TODO: standardize X
def get_data(file_path, is_test):
    df = pd.read_csv(file_path, sep=',', header=None)
    header = df.iloc[0][1:].as_matrix()
    encoded_y = 0

    if not is_test:
        y = df[1:][0].as_matrix()
        le = preprocessing.LabelEncoder()
        le.fit(y)
        encoded_y = le.transform(y)
        print(le.inverse_transform([0, 5]))

    # each column represent a data point
    if not is_test:
        X = df.drop(index=0, columns=0).as_matrix().astype(np.float).T
    else:
        X = df.drop(index=0).as_matrix().astype(np.float).T

    return X, encoded_y


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
