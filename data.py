import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

NUM_OF_BINS = 50

class Data:

    def __init__(self ,input_data):
        self.origin_data = input_data
        self.prob_data = None
        self.bins = None

    def generate_new_prob(self):
        def histogram(arr):
            return np.histogram(arr, bins=len(arr))

        k = np.apply_along_axis(histogram, 1, self.origin_data)
        counts, bins = np.apply_along_axis(histogram, 1, self.origin_data).T
        b = np.expand_dims(counts.T, axis=0)
        a = np.sum(b.T, axis=0)

        prob_mat = counts/np.expand_dims(np.sum(counts, axis=1).T, axis=0).T
        counts = counts.T / np.sum(counts.T, axis=0)
        a = 1
