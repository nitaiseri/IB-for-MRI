import numpy as np
from typing import List
from scipy.stats import norm
from load_data import *
import seaborn as sns
from fitter import Fitter
import scipy.stats
from matplotlib import pyplot as plt
import os
from numpy.random import normal
from numpy import hstack
from numpy import asarray
from numpy import exp



class ProbabilityMaker:
    """
    An abstract class to to calculate probability tables out of list of HumanScans
    """
    @staticmethod
    def smooth_histogram(points, means, num_of_add_noise=None):
        """
        given values of qMRI scans (can be mean of each area, all voxels of relevant areas,
        or all voxels and so on) calculate the probability to get the value of the mean of
        each area.
        :param points: nd array of values to consider.
        :param means: nd array of means of each area that we want to know the probability to get this
        particular value.
        :param num_of_add_noise: optional parameter, if we wand add noise to each point because we dont have
        enough, so we generate artificial point from normal distributions with mean of the value. so this is
        number of points we want to add for each point.
        :return: list of probabilities to get each mean we get in means(already normalize to 1)
        """
        #TODO: still missing. this fit is not good. need to replace with cumulative distribution function,
        # and integrate to each point.
        if num_of_add_noise:
            num_of_gaussians = len(points)
            sigma = np.ones(num_of_gaussians)*(np.mean(points)*0.05)
            values = (np.random.normal(size=(num_of_gaussians, num_of_add_noise)) * sigma[:, None] + points[:, None]).flatten()
        else:
            values = points
        fit = Fitter(list(values), distributions=[
                                        # 'gamma',
                                        # 'lognorm',
                                        # "beta",
                                        # "burr",
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
        """
        given list of subjects, parameter of scan and list of means of each area, calculate the
        probability table (areas over subjects) from generating points from those means.
        :param subjects: nd array subjects.
        :param parameter: parameter of scan
        :param means: nd array of means of each area that we want to know the probability to get this
        particular value.
        :return: 2d np table of the probability to get each the mean value of area given subject.
        """
        voxels = np.array([subject.get_all_voxels_per_areas(parameter) for subject in subjects])
        sigmas = np.array([np.std(voxel) for voxel in voxels])
        return np.array([ProbabilityMaker.smooth_histogram(subject_mean, subject_mean, 1000) for subject_mean in means])

    @staticmethod
    def voxels_in_areas_probability(subjects: List[HumanScans], parameter, means) -> np.array:
        """
         given list of subjects, parameter of scan and list of means of each area, calculate the
         probability table (areas over subjects) from all voxels in those areas.
         :param subjects: nd array subjects.
         :param parameter: parameter of scan
         :param means: nd array of means of each area that we want to know the probability to get this
         particular value.
         :return: 2d np table of the probability to get each the mean value of area given subject.
         """
        voxels = np.array([subject.get_all_voxels_per_areas(parameter) for subject in subjects])
        return np.array([ProbabilityMaker.smooth_histogram(voxels[i], means[i]) for i in range(len(means))])

    @staticmethod
    def total_voxels_probability(subjects: List[HumanScans], parameter, means) -> np.array:
        """
         given list of subjects, parameter of scan and list of means of each area, calculate the
         probability table (areas over subjects) from all voxels in the subject brain.
         :param subjects: nd array subjects.
         :param parameter: parameter of scan
         :param means: nd array of means of each area that we want to know the probability to get this
         particular value.
         :return: 2d np table of the probability to get each the mean value of area given subject.
         """
        voxels = np.array([subject.get_all_voxels(parameter) for subject in subjects])
        return np.array([ProbabilityMaker.smooth_histogram(voxels[i], means[i]) for i in range(len(means))])


def generate_data():
    """
    general function that ran through those three functions and generate probability table for each
    function, for each parameter of scan and save them.
    """
    #TODO: the save part not complete yet. dunno way but cannot save it with np.save.
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
            tables.append(func(subjects, parameter, means))

            # means_stds = subjects[0].get_mean_per_param(parameter).T
            #
            # prob = []
            # for i in range(means_stds[0].shape[0]):
            #     x = np.ones(means_stds[0].shape[0])*means_stds[0][i]
            #     mean = means_stds[0]
            #     std = means_stds[1]
            #     prob.append(np.sum(np.array(np.exp(-0.5*(x-mean)**2)*(mean-x)/(np.sqrt(2*np.pi*std)))))
            #
            #
            # x = means[0]
            # x_d = np.linspace(0.02, 0.06, 1000)
            # density = sum(norm(xi).pdf(x_d) for xi in x)
            #
            # plt.fill_between(x_d, density, alpha=0.5)
            # plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

            plt.axis([0.02, 0.06, -0.2, 5]);
        with open('raw_data/new/' + file_name + '.npy', 'wb') as f:
            np.save(f, np.array(tables))


def sum_gaussian(means, stds):
    # generate a sample
    samples = np.array([])
    for i in range(means.shape[0]):
        sample = normal(loc=means[i], scale=stds[i], size=1000)
        samples = hstack((samples, sample))
    points = np.linspace(samples.min(), samples.max(), num=100)
    values = np.array([np.sum(np.exp(-0.5*((point-means)/stds)**2)/(np.sqrt(2*np.pi)*stds)) for point in points])
    values = values/(np.sum(values)*((samples.max()-samples.min())/100))
    a = plt.hist(samples, bins=50, density=True)
    plt.plot(points, values)
    plt.show()


if __name__ == '__main__':
    # example of kernel density estimation for a bimodal data sample

    from sklearn.neighbors import KernelDensity

    # generate a sample
    # sample1 = normal(loc=20, scale=5, size=300)
    # sample2 = normal(loc=40, scale=5, size=700)
    # sample = hstack((sample1, sample2))
    # # fit density
    # model = KernelDensity(bandwidth=2, kernel='gaussian')
    # sample = sample.reshape((len(sample), 1))
    # model.fit(sample)
    # # sample probabilities for a range of outcomes
    # values = asarray([value for value in range(1, 60)])
    # values = values.reshape((len(values), 1))
    # probabilities = model.score_samples(values)
    # probabilities = exp(probabilities)
    # # plot the histogram and pdf
    # plt.hist(sample, bins=50, density=True)
    # plt.plot(values[:], probabilities)
    # plt.show()

    means = np.array([1, 3, 5])
    stds = np.ones(means.shape[0])/2
    sum_gaussian(means, stds)
    #
    # # plot the histogram
    # a = plt.hist(samples, bins=50, cumulative=False)
    # y = a[1].reshape((len(a[1]), 1))
    # plt.plot(y[:-1], a[0])
    # plt.show()
    # generate_data()
    # PROB_FUNC_OPTIONS = [(ProbabilityMaker.mean_probability_data, "per_mean"),
    #                       (ProbabilityMaker.voxels_in_areas_probability, "per_area"),
    #                        (ProbabilityMaker.total_voxels_probability, "total")]

    # PROB_FUNC_OPTIONS = [(lambda x, b: np.zeros(9).reshape((3,3)), "per_mean"),
    #                       (lambda x, b: np.zeros(9).reshape((3,3)), "per_area"),
    #                        (lambda x, b: np.zeros(9).reshape((3,3)), "total")]
    # subjects = [2, 2, 3]

    # for func, file_name in PROB_FUNC_OPTIONS:
    #
    #     with open('raw_data/new/' + file_name + '.npy', 'rb') as f:
    #         a = np.load(f)
    #         b=1