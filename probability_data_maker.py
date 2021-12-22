import numpy as np
import matplotlib.pyplot as plt
from typing import List
from load_data import *
import seaborn as sns
from fitter import Fitter
import scipy.stats
from matplotlib import pyplot as plt
import os


class ProbabilityMaker:
    @staticmethod
    def smooth_histogram(points, means, num_of_add_noise=None):
        if num_of_add_noise:
            num_of_gaussians = len(points)
            sigma = np.ones(num_of_gaussians)*(np.mean(points)*0.05)
            values = (np.random.normal(size=(num_of_gaussians, num_of_add_noise)) * sigma[:, None] + points[:, None]).flatten()
        else:
            values = points
        fit = Fitter(list(values), distributions=[
                                        'gamma',
                                        'lognorm',
                                        "beta",
                                        "burr",
                                        "norm"])
        fit.fit()
        # fit.summary()
        dist = scipy.stats.gamma
        param = fit.fitted_param['norm']
        pdf_fitted = dist.pdf(means, *param)
        return pdf_fitted/np.sum(pdf_fitted)
        # plt.plot(means, pdf_fitted, 'o-')
        # plt.show()

    @staticmethod
    def mean_probability_data(subjects: List[HumanScans], parameter, means) -> np.array:
        return np.array([ProbabilityMaker.smooth_histogram(subject_mean, subject_mean, 1000) for subject_mean in means])

    @staticmethod
    def voxels_in_areas_probability(subjects: List[HumanScans], parameter, means) -> np.array:
        voxels = np.array([subject.get_all_voxels_per_areas(parameter) for subject in subjects])
        return np.array([ProbabilityMaker.smooth_histogram(voxels[i], means[i]) for i in range(len(means))])

    @staticmethod
    def total_voxels_probability(subjects: List[HumanScans], parameter, means) -> np.array:
        voxels = np.array([subject.get_all_voxels(parameter) for subject in subjects])
        return np.array([ProbabilityMaker.smooth_histogram(voxels[i], means[i]) for i in range(len(means))])


def generate_data():
    PROB_FUNC_OPTIONS = [(ProbabilityMaker.mean_probability_data, "per_mean"),
                          (ProbabilityMaker.voxels_in_areas_probability, "per_area"),
                           (ProbabilityMaker.total_voxels_probability, "total")]

    subjects = load_all_subjects()

    if not os.path.exists('raw_data/new'):
        os.makedirs('raw_data/new')

    for func, file_name in PROB_FUNC_OPTIONS:
        tables = []
        for parameter in PARAMETERS:
            means = np.array([subject.get_mean_per_param(parameter) for subject in subjects])
            voxels = np.array([subject.get_all_voxels_per_areas(parameter) for subject in subjects])
            a = np.mean(means)
            b = a * 0.05
            c = np.std(means)
            d = np.array([np.std(voxel) for voxel in voxels])
            tables.append(func(subjects, parameter, means))
        with open('raw_data/new/' + file_name + '.npy', 'wb') as f:
            np.save(f, np.array(tables))



if __name__ == '__main__':

    # PROB_FUNC_OPTIONS = [(lambda x, b: np.zeros(9).reshape((3,3)), "per_mean"),
    #                       (lambda x, b: np.zeros(9).reshape((3,3)), "per_area"),
    #                        (lambda x, b: np.zeros(9).reshape((3,3)), "total")]
    # subjects = [2, 2, 3]

    for func, file_name in PROB_FUNC_OPTIONS:

        with open('raw_data/new/' + file_name + '.npy', 'rb') as f:
            a = np.load(f)
            b=1