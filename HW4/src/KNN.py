import csv
import math
import random


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open('../data/{}'.format(filename), 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


# training_set = []
# test_set = []
#
# loadDataset('iris.data', 0.66, training_set, test_set)
# print('training set: '+ str(len(training_set)))
# print('test set: '+ str(len(test_set)))


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print('Distance: ' + str(distance))
