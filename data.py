from builtins import print

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import load_data

NUM_OF_BINS = 50


class Data:
    @staticmethod
    def histogram(arr):
        return np.histogram(arr, len(np.unique(arr)) - 1)

    @staticmethod
    def generate_new_prob(input_data):
        counts, bins = np.apply_along_axis(Data.histogram, 1, input_data).T
        probability_matrix = []
        for raw in counts:
            probability_matrix.append(raw/np.sum(raw, axis=0))
        return probability_matrix, bins
        # prob_mat = counts/np.expand_dims(np.sum(counts, axis=1).T, axis=0).T
        # counts = counts.T / np.sum(counts.T, axis=0)
        # a = 1

if __name__ == '__main__':
    mu = np.random.choice([2, 3, 5, 1], 10)
    sigma = np.random.choice([2, 1, 3, 1], 10)

    a = np.random.normal(size=(10, 10)) * sigma[:, None] + mu[:, None]


