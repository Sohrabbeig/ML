import numpy as np
from scipy.special import comb
from matplotlib import pyplot as plt
from math import *


def special_mult(arr):
    temp = 0
    for i in range(len(arr)):
        temp += sum(arr[i] * arr[i + 1:])
    return temp


class K_Means:
    def __init__(self, X, y, tol=0.001, max_iter=300):
        self.y = y
        self.tol = tol
        self.max_iter = max_iter
        self.clusters = {}
        purity_list = []
        rand_index_list = []
        mutual_information_list = []
        iters = []

        for k in range(2, 11):
            self.fit(X, k)
            iters.append(k)
            purity_list.append(self.purity())
            rand_index_list.append(self.rand_index())
            mutual_information_list.append(self.mutual_information(k))

        plt.plot(iters, purity_list)
        plt.xlabel("iteration number")
        plt.ylabel("purity")
        plt.show()
        plt.plot(iters, rand_index_list)
        plt.xlabel("iteration number")
        plt.ylabel("rand_index")
        plt.show()
        plt.plot(iters, mutual_information_list)
        plt.xlabel("iteration number")
        plt.ylabel("mutual information")
        plt.show()

    def fit(self, data, k):

        self.centroids = {}

        for i in range(k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.clusters = {}

            for i in range(k):
                self.clusters[i] = []

            for feature_set in data:
                distances = [np.linalg.norm(
                    feature_set-self.centroids[centroid]) for centroid in self.centroids]
                c = distances.index(min(distances))
                self.clusters[c].append(feature_set)

            prev_centroids = dict(self.centroids)

            for c in self.clusters:
                self.centroids[c] = np.average(
                    self.clusters[c], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.linalg.norm(original_centroid-current_centroid) > self.tol:
                    optimized = False
                    break

            if optimized:
                for i in range(k):
                    self.clusters[i] = []

                for i in range(len(data)):
                    distances = [np.linalg.norm(
                        data[i]-self.centroids[centroid]) for centroid in self.centroids]
                    c = distances.index(min(distances))
                    self.clusters[c].append(self.y[i])

                break

    def purity(self):
        total = 0
        m = 0

        for i in self.clusters.values():
            m += np.max(np.unique(i, return_counts=True)[1])
            total += len(i)

        purity = m / total

        return purity

    def rand_index(self):
        P = sum([comb(len(x), 2, exact=True) for x in self.clusters.values()])
        TP = 0
        total = comb(len(self.y), 2, exact=True)
        n_true_clusters = len(np.unique(self.y))
        partitions = []
        for i in range(n_true_clusters):
            partitions.append([])

        for i in self.clusters.values():
            for j in np.unique(i, return_counts=True)[1]:
                if j > 1:
                    TP += comb(j, 2, exact=True)

        FP = P - TP

        for i in self.clusters.values():
            temp = np.unique(i, return_counts=True)
            for j in range(len(temp[0])):
                partitions[temp[0][j]].append(temp[1][j])

        FN = sum(map(special_mult, partitions))

        TN = total - (TP + FP + FN)

        return (TP + TN) / total

    def mutual_information(self, k):
        P = np.zeros([k, len(np.unique(self.y))])
        N = len(self.y)
        U = np.array([len(x) for x in self.clusters.values()]) / N
        V = np.unique(self.y, return_counts=True)[1] / N
        I = 0

        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                P[i][j] = sum(np.array(self.clusters[i]) == j) / N
                try:
                    I += P[i][j] * log(P[i][j] / (U[i] * V[j]))
                except:
                    pass

        return I
