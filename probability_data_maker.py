import numpy as np
from typing import List
from scipy.stats import norm
from load_data import *
import seaborn as sns
import scipy.stats
from matplotlib import pyplot as plt
import os
from numpy.random import normal
from numpy import hstack
from numpy import asarray
from numpy import exp
import random
import load_data
from Consts import *
from config import *
import probability_data_maker as pdm



class ProbabilityMaker:
    """
    An abstract class to to calculate probability tables out of list of HumanScans
    """
    @staticmethod
    def get_means_std(subjects, parameter):
        """
        Calculate and return list of means and stds per area.
        :param subjects: The subjects to calculate over.
        :param parameter: the parameter the calculation work on.
        :return: 2 lists of 2 dim nd array. for each subject, array per area.
        """
        means = [] 
        stds = []
        for subject in subjects:
            mean, std = subject.get_mean_std_per_param(parameter)
            means.append(mean)
            stds.append(std)
        return np.array(means), np.array(stds)

    ### The next three function are 3 methods to calculate the probability function of qMRI values. ###
    ### We check three of them and examined the outcome. Bottom line, we need only one of them,     ###
    ### and we not sure, but it seems to be that the second one  'voxels_in_areas_probability'      ###
    ### look that works better.                                                                     ###

    @staticmethod
    def mean_probability_data(subjects, parameter, means, path) -> np.array:
        """
        given list of subjects, parameter of scan and list of means of each area, calculate the
        probability table (areas over subjects) from generating points from those means.
        :param subjects: nd array subjects.
        :param parameter: parameter of scan
        :param means: nd array of means of each area that we want to know the probability to get this
        particular value.
        :return: 2d np table of the probability to get each the mean value of area given subject.
        """
        prob_table = []
        means, stds = ProbabilityMaker.get_means_std(subjects, parameter)
        for i in range(len(means)):
            prob_table.append(sum_gaussian(means[i], stds[i],means[i], PLOT, "per_mean "+parameter, path))

        prob_table = np.array(prob_table)
        return prob_table/prob_table.sum(axis=1)[:, np.newaxis]

    @staticmethod
    def voxels_in_areas_probability(subjects, parameter, means, path) -> np.array:
        """
         given list of subjects, parameter of scan and list of means of each area, calculate the
         probability table (areas over subjects) from all voxels in those areas.
         :param subjects: nd array subjects.
         :param parameter: parameter of scan
         :param means: nd array of means of each area that we want to know the probability to get this
         particular value.
         :return: 2d np table of the probability to get each the mean value of area given subject.
         """
        prob_table = []
        params = np.array([subject.get_all_voxels_per_areas(parameter) for subject in subjects])
        for i in range(params.shape[0]):
            filteres_means, filtered_stds = filter_values(params[i][0], params[i][1])
            prob_table.append(sum_gaussian(filteres_means, filtered_stds , means[i], PLOT, "per_area "+parameter, path))
        prob_table = np.array(prob_table)
        return prob_table/prob_table.sum(axis=1)[:, np.newaxis]

    @staticmethod
    def total_voxels_probability(subjects, parameter, means, path) -> np.array:
        """
         given list of subjects, parameter of scan and list of means of each area, calculate the
         probability table (areas over subjects) from all voxels in the subject brain.
         :param subjects: nd array subjects.
         :param parameter: parameter of scan
         :param means: nd array of means of each area that we want to know the probability to get this
         particular value.
         :return: 2d np table of the probability to get each the mean value of area given subject.
         """
        prob_table = []
        voxels = np.array([subject.get_all_voxels(parameter) for subject in subjects])
        for i in range(len(subjects)):
            voxs = filter_values(voxels[i])
            stds = np.ones(voxs.shape[0])*np.mean(voxs) * 0.03
            num_of_voxs = voxs.shape[0]
            # prob_table.append(get_gaussian_pdf(voxels[i], np.ones(num_of_voxs)*np.std(voxels[i]), means[i]))
            prob_table.append(sum_gaussian(voxs, stds, means[i], PLOT, "total "+parameter, path))

        prob_table = np.array(prob_table)
        return prob_table/prob_table.sum(axis=1)[:, np.newaxis]

def filter_values(vals, optional_vals=np.array([0])):
    """
    filter values of voxels which is too strange. mean, filter out those who 4std away from the mean.
    :param vals: the values to filter.
    :param optional_vals: more vals to filter.
    :return: the filtered values.
    """
    mask = (vals < (np.mean(vals)+4*np.std(vals))) & (vals > 0) & (vals > (np.mean(vals)-4*np.std(vals)))
    if optional_vals.any():
        return vals[mask], optional_vals[mask]
    return vals[mask]


def generate_data(subjects=None, sub_path=None):
    """
    general function that ran through those three functions and generate probability table for each
    function, for each parameter of scan and save them.
    """
    if RUN:
        pk_generate_data(subjects, sub_path)
    else:
        path = RAW_DATA_PATH + sub_path
        if not subjects:
            subjects = load_all_HUJI_subjects()
            
            # The path to where save the data and plots.
            path = '/no_name/'
        
            # save data of subject and areas for the IB
            save_data(subjects, path)
        
        # Loop over function
        for func, file_name in PROB_FUNC_OPTIONS:
            if not os.path.exists(path + file_name + '/'):
                os.makedirs(path + file_name + '/')
            # Loop over prameters
            for parameter in PARAMETERS:
                means, stds = ProbabilityMaker.get_means_std(subjects, parameter)
                table = func(subjects, parameter, means, path)
                with open(path + file_name + '/' + parameter + '.npy', 'wb') as f:
                    np.save(f, table)
                print(file_name + " " + parameter + " Done!\n")

def pk_generate_data(subjects=None, path=None):
    """
    general function that ran through those three functions and generate probability table for each
    function, for each parameter of scan and save them.
    """
    # Loop over function
    path = RAW_DATA_PATH + path
    for func, file_name in PROB_FUNC_OPTIONS[1:-1]:
        if not os.path.exists(path + file_name + '/'):
            os.makedirs(path + file_name + '/')
        # Loop over prameters
        for parameter in PARAMETERS:
            if subjects[0].parameters[parameter] is None:
                continue
            means, stds = pdm.ProbabilityMaker.get_means_std(subjects, parameter)
            table = func(subjects, parameter, means, path)
            with open(path + file_name + '/' + parameter + '.npy', 'wb') as f:
                np.save(f, table)
            print(file_name + " " + parameter + " Done!\n")


def get_gaussian_pdf(means, stds, points):
    """
    calculate the pdf function of normal distribution define by means and std and return
    the values at specific points.
    @param means: nd_array of single value of mean(for distribution).
    @param stds: nd_array of single value of std(for distribution).
    @param points: the points to calculate the value of the calculated pdf at those points.
    @return: list of pdf values of the sum of normal distributions at points. (or single value in case of single point).
    """
    if type(points).__module__ != np.__name__ and not isinstance(points, list):
        return np.sum(np.exp(-0.5 * ((points - means) / stds) ** 2) / (np.sqrt(2 * np.pi) * stds))
    return np.array([np.sum(np.exp(-0.5 * ((point - means) / stds) ** 2) / (np.sqrt(2 * np.pi) * stds)) for point in points])

def get_gaussian_cdf(means, stds, points):
    """
    calculate the cdf function of normal distribution define by means and std and return
    the values at specific points.
    @param means: nd_array of single value of mean(for distribution).
    @param stds: nd_array of single value of std(for distribution).
    @param points: the points to calculate the value of the calculated pdf at those points.
    @return: list of pdf values of the sum of normal distributions at points. (or single value in case of single point).
    """
    pdf_points, pdf_values = sum_gaussian(means, stds)
    cdf_values = np.cumsum(pdf_values/np.sum(pdf_values))
    cdf_values = np.concatenate([[0], cdf_values])
    if type(points).__module__ != np.__name__ and not isinstance(points, list):
        return cdf_values[np.digitize(points, pdf_points)]
    indicies = np.digitize(points, pdf_points)
    return np.array([cdf_values[index] for index in indicies])

def sum_gaussian(means, stds, vals, plot=False, title = None, path = None):
    """
    This function calaulate the probabilities to get each value in vals.
    by sum up the gaussians the defined by the means and the stds.
    If the means and stds are not too much so generate more by sampleing.
    :param means: Array of the means of the gaussians.
    :param stds: Array of the means of the gaussians.
    :param vals: Array of the means of the gaussians.
    :param plot: Boolean if to plot the histogram and the porobability curve.
                    Used to see if the normalization is good.
    :param title: The title of the plot above.
    :param path: Path to save the plot if generated.
    :return: List of probability values the represent the probability to get 
            the val in vals depends on the means and stds.
    """
    # generate a sample if necessary.
    samples = np.array([])
    if len(means) < 100 :
        num_of_delta_x = 100
        for i in range(means.shape[0]):
            sample = normal(loc=means[i], scale=stds[i], size=1000)
            samples = hstack((samples, sample))
    else:
        num_of_delta_x = 100
        samples = means
    # manual calculate sum of exp
    points = np.linspace(samples.min(), samples.max(), num=num_of_delta_x)
    values = get_gaussian_pdf(means, stds, points)
    data_vals = get_gaussian_pdf(means, stds, vals)
    values = values/(np.sum(values)*((samples.max()-samples.min())/num_of_delta_x))

    if plot:
        hist_values, bins, other = plt.hist(samples, bins=75, density=True)
        # values_2 = get_gaussian_cdf(means, stds, points)
        plt.plot(points, values)
        # plt.plot(points, values_2)
        plt.title(title)
        plt.savefig(path + '/' + title)
        plt.show()
    return data_vals

