from math import *
import pandas as pd
import numpy as np


# TODO: compare with sklearn pca decomposition
def PCA(X):

    def find_k(sorted_eig_vals):
        k = 0
        s = 0
        total = sum([x ** 2 for x in sorted_eig_vals])

        while s / total < 0.999:
            s += sorted_eig_vals[k] ** 2
            k += 1

        return k

    cov_mat = np.cov(X)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i])
                 for i in range(len(eig_val_cov))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    k = find_k(sorted(eig_val_cov, reverse=True))
    selected_eig_vecs = [x[1].reshape(len(X), 1) for x in eig_pairs[:k]]
    matrix_w = np.hstack(selected_eig_vecs)
    transformed = matrix_w.T.dot(X)

    return transformed


def LDA(X):
    pass
