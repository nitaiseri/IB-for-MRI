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

        sns.hisplot
        return probability_matrix, bins
        # prob_mat = counts/np.expand_dims(np.sum(counts, axis=1).T, axis=0).T
        # counts = counts.T / np.sum(counts.T, axis=0)
        # a = 1




# xs = randrange(n, 23, 32)
# ys = randrange(n, 0, 100)
# zs = randrange(n, 0, 100)
measure, seg = load_data.main()
a = np.unique(seg)
b = 1


