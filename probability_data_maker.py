import numpy as np
from typing import List
from load_data import *
import seaborn as sns
from fitter import Fitter
import scipy.stats
from matplotlib import pyplot as plt
import os
STD = 0.005


class ProbabilityMaker:
    @staticmethod
    def smooth_by_gaussian(points, means, nums):
        # values = np.array([])
        # for point in points:
        #     # values += list(np.random.normal(point, STD, nums))
        #     values = np.concatenate((values, np.random.normal(point, STD, nums)))
        num_of_gaussians = len(points)
        sigma = np.ones(num_of_gaussians)*STD
        values = (np.random.normal(size=(num_of_gaussians, nums))*sigma[:, None] + points[:, None]).flatten()
        f = Fitter(list(values), distributions=[
                                        # 'gamma',
                                        # 'lognorm',
                                        # "beta",
                                        # "burr",
                                        "norm"])
        f.fit()
        # f.summary()
        dist = scipy.stats.gamma
        param = f.fitted_param['norm']
        pdf_fitted = dist.pdf(means, *param)
        return pdf_fitted/np.sum(pdf_fitted)
        # plt.plot(means, pdf_fitted, 'o-')
        # plt.show()

    @staticmethod
    def mean_probability_data(subjects: List[HumanScans], parameter) -> np.array:
        means = np.array([subject.get_mean_per_param(parameter) for subject in subjects])
        return np.array([ProbabilityMaker.smooth_by_gaussian(subject_mean, subject_mean,  1000) for subject_mean in means])

    @staticmethod
    def voxels_in_areas_probability(subjects: List[HumanScans], parameter) -> np.array:
        means = np.array([subject.get_mean_per_param(parameter) for subject in subjects])
        voxels = np.array([subject.get_all_voxels_per_areas(parameter) for subject in subjects])
        return np.array([ProbabilityMaker.smooth_by_gaussian(voxels[i], means[i], 100) for i in range(len(means))])

    @staticmethod
    def total_voxels_probability(subjects: List[HumanScans], parameter) -> np.array:
        means = np.array([subject.get_mean_per_param(parameter) for subject in subjects])
        voxels = np.array([subject.get_all_voxels(parameter) for subject in subjects])
        return np.array([ProbabilityMaker.smooth_by_gaussian(voxels[i], means[i], 10) for i in range(len(means))])


if __name__ == '__main__':
    PROB_FUNC_OPTIONS = [(ProbabilityMaker.mean_probability_data, "per_mean"),
                          (ProbabilityMaker.voxels_in_areas_probability, "per_area"),
                           (ProbabilityMaker.total_voxels_probability, "total")]

    subjects = load_all_subjects()
    # PROB_FUNC_OPTIONS = [(lambda x, b: x[0], "per_mean"),
    #                       (lambda x, b: x[1], "per_area"),
    #                        (lambda x, b: x[2], "total")]
    # subjects = [2, 2, 3]
    if not os.path.exists('raw_data/new'):
        os.makedirs('raw_data/new')
    for func, file_name in PROB_FUNC_OPTIONS:
        tables = []
        for parameter in PARAMETERS:
             tables.append(func(subjects, parameter))
        with open('raw_data/new/' + file_name + '.npy', 'wb') as f:
            np.save(f, np.array(tables))

    # for func, file_name in PROB_FUNC_OPTIONS:
    #
    #     with open('raw_data/new/' + file_name + '.npy', 'rb') as f:
    #         a = np.load(f)
    #         b=1