import pandas as pd
import numpy as np


def get_data(file_path):
    df = pd.read_csv(file_path, sep=',', header=None)
    header = df.iloc[0][1:].as_matrix()
    y = df[1:][0].as_matrix()
    # each column represent a data point
    X = df.drop(index=0, columns=0).as_matrix().astype(np.float).T

    return X
