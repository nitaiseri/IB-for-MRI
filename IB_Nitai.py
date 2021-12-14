import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.cluster import hierarchy
import matplotlib
import pickle
from data import Data

CLUSTERS = ["MTsat", "R1", "MD", "R2", "MTV", "R2s"]
# CLUSTERS = ["MTV", "R2s"]
CLUSTERS_DIC = {"MTsat": 0, "R1": 1, "MD": 2, "R2": 3, "MTV": 4, "R2s": 5}
NUM_OF_BETA = 700
END_NAME = "one_over"
SOURCE_DIR = END_NAME + "/"
MTsat, R1, MD, R2, MTV, R2s = 0, 1, 2, 3, 4, 5
ANALYSE_BY_AREAS = True
ANALYSE_BY_PEOPLE = False
ANALYSE_TYPE = "ANALYSE_BY_AREAS"
NORMALIZATION = {"square": lambda x: np.square(x),
                 "exp": lambda x: np.exp(-1/x),
                 "one_over": lambda x: np.square(1/x)}


def generate_beta(max_value=20000, length=NUM_OF_BETA):
    """
    function to generate list of decreasing beta's
    :param max_value: first beta value
    :param length: num of beta's
    :return: list of the beta's
    """
    beta_values = [max_value]
    for idx in range(length-1):
        beta_values.append(beta_values[-1]*0.99)
    return beta_values


def D_kl(p1, p2):
    """
    calculate the kl distance between two matrices.
    :param p1: matrix 1.
    :param p2: matrix 2.
    :return: matrix of kl distances.
    """
    C1 = np.einsum('ij,ik->ijk', p1, 1 / p2)  ###is it the right way?
    C2 = np.log(C1)
    C2[p1 == 0, :] = 0
    return np.einsum('ij,ijk->jk', p1, C2)


def D_kl_vec(p_y_x_hat):
    """
    calculate kind of projecton of
    :param p_y_x_hat:
    :return:
    """
    p_mean = np.mean(p_y_x_hat, axis=1)
    return np.apply_along_axis(lambda x: np.sum(np.where(x > 0, x * np.log(x / p_mean), 0)), 1, p_y_x_hat.T)


def load_data(path):
    """
    specific function to load and preprocess first data(shir's)
    :param path: path to .mat data
    :return: input matrices, subjects, region, area_names, area_types
    """
    x = loadmat(path)
    mean_values = {}
    subjects = {}
    region = {}

    with open('raw_data/gender.txt') as f:
        gender = f.read().splitlines()
    with open('raw_data/age.txt') as f:
        age = f.read().splitlines()
    with open('raw_data/area_names2.txt') as f:
        area_names = f.read().splitlines()
    with open('raw_data/area_types.txt') as f:
        area_types = f.read().splitlines()

    subj_id = [str(x + 1) for x in range(45)]
    sub = np.array(list(zip(subj_id, age, gender)))

    for ix in range(6):
        subjects[ix] = sub.copy()
        mean_values[ix] = x['huji_data']['data'][0][0][:, :, ix]

        mean_values[ix] = np.delete(mean_values[ix], 29, 0)  # remove Left Accumbens
        mean_values[ix] = np.delete(mean_values[ix], 20, 0)  # remove Medulla
        mean_values[ix] = np.delete(mean_values[ix], 8, 0)  # remove Right Accumbens

        subjects[ix] = subjects[ix][~np.isnan(mean_values[ix].T).any(axis=1)]
        mean_values[ix] = (mean_values[ix].T[~np.isnan(mean_values[ix].T).any(axis=1)]).T

    del area_names[29]  # remove Left Accumbens
    del area_names[20]  # remove Medulla
    del area_names[8]  # remove Right Accumbens

    del area_types[29]  # remove Left Accumbens
    del area_types[20]  # remove Medulla
    del area_types[8]  # remove Right Accumbens
    for x, y in zip(area_names, area_types):
        region[x] = y
    return mean_values, subjects, region, area_names, area_types


def normalize_data(input_matrix, normalization=None):
    """
    function that normalize the input data.
    :param input_matrix: the data to normalize.
    :param normalization: type of normalization.
    :return: normalize data.
    """
    if normalization:
        return NORMALIZATION[normalization](input_matrix)
    return input_matrix


class IB:
    """
    class for MRI's data for the iterative IB algorithm.
    """
    def __init__(self, input_matrix, subjects, regions, area_names, area_types, beta_values, name_of_scan, normalization=None):
        self.beta_max = None
        self.name_of_scan = name_of_scan
        self.beta_values = beta_values
        self.analyse_by_areas = ANALYSE_BY_AREAS
        self.region = regions
        self.subjects = subjects
        self.input_matrix = normalize_data(input_matrix, normalization)
        self.p_y_x_hat = None
        self.p_x_given_x_hat = None
        self.clusters_matrix = []
        self.full_distances = []
        self.clus = []
        self.area_names = area_names
        self.area_types = area_types

    def IB_iter(self, p_x, p_y_x, p_x_hat_given_x, beta):
        """
        calculate new P(x^|x) according to IB formula
        :param p_x: P(x)
        :param p_y_x: p(y|x)
        :param p_x_hat_given_x: the former P(x^|x)
        :param beta: the current beta value
        :return: the new P(x^|x)
        """
        p_x_hat = p_x_hat_given_x @ p_x
        p_x_given_x_hat = (p_x_hat_given_x * p_x).T / p_x_hat
        p_y_x_hat = p_y_x @ p_x_given_x_hat
        not_norm = np.exp(-beta * D_kl(p_y_x, p_y_x_hat)) * p_x_hat
        not_norm = not_norm.T
        return not_norm / np.sum(not_norm, axis=0)

    def prepare_prob(self, input_matrix):
        """
        Initialize probability matrix P(y|x), P(x^|x) and vector P(x)
        :param input_matrix: P(y|x) not normalized
        :return: P(y|x) normalized, P(x^|x) ,P(x)
        """
        p_y_x = input_matrix / np.sum(input_matrix, axis=0)

        x_dim = p_y_x.shape[1]
        p_x = np.ones(x_dim)
        p_x = p_x / np.sum(p_x)

        p_x_hat_given_x = np.eye(x_dim) + abs(np.random.normal(0, 0.02, (x_dim, x_dim)))
        p_x_hat_given_x = p_x_hat_given_x / np.sum(p_x_hat_given_x, axis=0)

        return p_y_x, p_x, p_x_hat_given_x

    def get_clusters(self):
        """
        main function of running the iterative IB  algorithm over the data.
        :return:None
        """
        p_y_x, p_x, p_x_hat_given_x = self.prepare_prob(self.input_matrix)

        for beta in self.beta_values:
            err = 1
            index = 0
            if np.any(np.where(p_x_hat_given_x < np.exp(-200))):
                p_x_hat_given_x += np.exp(-100)
                p_x_hat_given_x = p_x_hat_given_x / np.sum(p_x_hat_given_x, axis=0)
            while err > (1 / beta) / 10:
                prev_p = p_x_hat_given_x
                p_x_hat_given_x = self.IB_iter(p_x, p_y_x, p_x_hat_given_x, beta)
                err = np.sum(abs(prev_p - p_x_hat_given_x))
                index += 1

            p_x_hat = p_x_hat_given_x @ p_x
            p_x_given_x_hat = (p_x_hat_given_x * p_x).T / p_x_hat
            if np.any(np.isnan(p_x_given_x_hat)):
                breakpoint()

            p_y_x_hat = p_y_x @ p_x_given_x_hat
            self.full_distances.append(D_kl_vec(p_y_x_hat))
            self.clus.append(np.linalg.matrix_rank(p_x_given_x_hat, tol=(1 / beta) / 10))
            t, indices = np.unique(p_x_given_x_hat.round(decimals=int(np.ceil(np.log10(10 * beta)))), axis=1,
                                   return_inverse=True)
            self.clusters_matrix.append(indices)
        self.clusters_matrix = np.array(self.clusters_matrix)
        self.p_y_x_hat = p_y_x_hat
        self.p_x_given_x_hat = p_x_given_x_hat
        # bad_betas = self.find_more_then_one()
        # if len(bad_betas)>1:
        #     self.beta_values = re_generate_beta(self.beta_values, bad_betas)
        #     self.clusters_matrix = []
        #     self.full_distances = []
        #     self.clus = []
        #     self.get_clusters()

    def run_analysis(self, which_ax=ANALYSE_BY_AREAS, name_of_file=None):
        """
        decide over which axes run the algorithm and run get_clusters, and save the object after analyse the data.
        :param name_of_file: name of the object file to save.
        :param which_ax: boolean that represent over which axes the algorithm gonna run.
        :return: None
        """
        self.analyse_by_areas = which_ax
        if self.analyse_by_areas:
            self.input_matrix = self.input_matrix.T
        if not self.beta_values:
            self.beta_values = generate_beta(self.find_beta_max(), NUM_OF_BETA)
            self.beta_max = self.beta_values[0]
        # else:
        #     self.beta_values = generate_beta(self.beta_max, NUM_OF_BETA)
        self.analyse_by_areas = which_ax
        self.get_clusters()
        if not name_of_file:
            name_of_file = self.name_of_scan
        filename = "data/" + SOURCE_DIR + ANALYSE_TYPE + "/" + name_of_file + "-" + ANALYSE_TYPE
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as ib_data_after_analysis:
            pickle.dump(self, ib_data_after_analysis)
        print(str(self.name_of_scan) + "-Done")

    def find_beta_max(self):
        beta = 2000
        while True:
            p_y_x, p_x, p_x_hat_given_x = self.prepare_prob(self.input_matrix)
            err = 1
            index = 0
            while err > (1 / beta):
                prev_p = p_x_hat_given_x
                p_x_hat_given_x = self.IB_iter(p_x, p_y_x, p_x_hat_given_x, beta)
                err = np.sum(abs(prev_p - p_x_hat_given_x))
                index += 1

            p_x_hat = p_x_hat_given_x @ p_x
            p_x_given_x_hat = (p_x_hat_given_x * p_x).T / p_x_hat

            rank = np.linalg.matrix_rank(p_x_given_x_hat, tol=(1 / beta) / 10)
            if rank == self.input_matrix.shape[1]:
                return beta
            beta += 500

    def find_more_then_one(self):
        bad_betas = []
        for idx in range(len(self.beta_values)-1):
            if self.clus[idx]-self.clus[idx+1] > 1:
                bad_betas.append(idx)
        return bad_betas


# Plot results:
def subj_to_text(subj):
    """
    make the subject into a string
    mainly for visualisation
    :param subj: the subject name
    :return: string of it
    """
    return 'Age: ' + subj[1] + ' (' + subj[2] + ')'


def plot_convergence_Dkl(ib_d):
    """
    visualisation plot to see convergence of kl distance of each cluster from mean over beta.
    :param ib_d: the IB object after analysed
    :return: None, save the plot in the same directory
    """

    plt.figure(figsize=(30,30))
    plt.rcParams["figure.figsize"]=30,30
    fd = np.array(ib_d.full_distances)
    try:
        plot_axis = [(max(ib_d.beta_values)*(0.99)**(np.where(~ib_d.clusters_matrix.any(axis=1))[0][0])) - 50,
                                                                    max(ib_d.beta_values), 0, fd.max() + 0.0002]
    except(IndexError):
        print(str(ib_d.name_of_scan) + "-" + END_NAME + " didn't converged to one cluster", file=sys.stderr)
        return None
    for idx in range(fd.shape[1]):
        plt.plot(ib_d.beta_values, fd[:, idx], linewidth=2)
    # if idx > fd.shape[1] / 2:
    #     plt.plot(ib_d.beta_values, -fd[:, idx], linewidth=2)
    # else:
    #     plt.plot(ib_d.beta_values, fd[:, idx], linewidth=2)

    #plot legend
    for idx in range(ib_d.p_y_x_hat.shape[1]):
        # text_condition = fd[0,idx] < plot_axis[3]
        text_condition = True
        if not(fd[0,idx] == np.inf):
            if text_condition:
                if ib_d.analyse_by_areas:
                    plt.text(plot_axis[1]*1.05, fd[0,idx],ib_d.area_names[idx],fontsize=20,rotation=45,rotation_mode = "anchor")
                else:
                    plt.text(plot_axis[1]*1.05, fd[0,idx],subj_to_text(ib_d.subjects[idx]),fontsize=20,rotation=45,rotation_mode = "anchor")

    plt.title(ib_d.name_of_scan + "-" + END_NAME, fontsize=20)
    plt.xscale("log")
    plt.xlabel('beta', fontsize=20)
    plt.ylabel('Dkl to mean', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.axis(plot_axis)
    filename = "plots/" + SOURCE_DIR + ANALYSE_TYPE + "/" + "Dkl_convergence/" + str(ib_d.name_of_scan) + ANALYSE_TYPE + "-" + END_NAME + ".png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    return 1


def load_analysed_data(name_of_file) -> IB:
    """
    load analysed data to IB object.
    :return: IB object of analysed data.
    """
    with open(name_of_file, 'rb') as ib_data:
        ib_data = pickle.load(ib_data)
    return ib_data


def pre_pros_for_hierarchy(ib_data):
    print(ib_data.name_of_scan)

    def find_multi(tt, x):
        return [i for i, y in enumerate(tt) if y == x]

    tt = ib_data.clusters_matrix
    num = tt.shape[1]
    idx_lst = list(range(num))
    cntr = num
    Z = []
    running_idx = 1

    def clus_iter(tt, cntr, idx_lst, Z, running_idx):
        for idx in range(len(ib_data.beta_values)):
            pp = list(tt[idx])
            for x in set(pp):
                if list(pp).count(x) > 1 and x != -1:
                    idxs = find_multi(pp, x)
                    if len(idxs) > 2:
                        print('oops', idxs)
                    tt[:, idxs[0]] = -1
                    # print('merge ' + str(idx_lst[idxs[1]]) + ' and ' + str(idx_lst[idxs[0]]) + ' into ' + str(cntr))
                    Z.append([idx_lst[idxs[1]], idx_lst[idxs[0]], idx, running_idx])
                    idx_lst[idxs[1]] = cntr
                    idx_lst[idxs[0]] = cntr
                    cntr += 1
                    running_idx += 1

        return tt, cntr, idx_lst, Z, running_idx

    tt, cntr, idx_lst, Z, running_idx = clus_iter(tt, cntr, idx_lst, Z, running_idx)

    Z = np.array(Z, dtype='double')
    Z[:, 2] = Z[:, 2] - min(Z[:, 2])

    # for i in range(Z.shape[0]):
    #     Z[i, 2] = i * 5

    return Z


def plot_hierarchy(ib_data, Z):
    matplotlib.rcParams['lines.linewidth'] = 5

    fig, ax = plt.subplots(1, 1)
    ax.set_title(ib_data.name_of_scan + "-" + END_NAME, fontsize=30)
    fig.set_size_inches(16, 8)
    # dn = hierarchy.dendrogram(Z,labels=area_names,leaf_rotation=-80, ax=ax)
    if ib_data.analyse_by_areas:
        dn = hierarchy.dendrogram(Z, labels=ib_data.area_names, ax=ax, orientation='right', color_threshold=160,
                                  above_threshold_color='k')
    else:
        # dn = hierarchy.dendrogram(Z,labels=[subj_to_text(x) for x in subjects[contrast]],orientation = 'right',leaf_rotation=-80, ax=ax)
        dn = hierarchy.dendrogram(Z, labels=np.array([subj_to_text(x) for x in ib_data.subjects]),
                                  ax=ax, leaf_rotation=-80, color_threshold=205, above_threshold_color='k')
    ax.tick_params(axis='y', which='major', labelsize=13)
    plt.tight_layout()

    hierarchy.set_link_color_palette(['#993404', '#64ad30', '#a2142e', '#7e2f8e'][::-1])
    filename = "plots/" + SOURCE_DIR + ANALYSE_TYPE + "/" + "hierarchy/" + str(ib_data.name_of_scan) + ANALYSE_TYPE + "-" + END_NAME + ".png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    # plt.show()


def main_analyze(analys_by, normal=None, beta_max=None):
    input_matrixes, subjects, regions, area_names, area_types = \
        load_data('raw_data/huji_data.mat')
    beta_values = generate_beta(beta_max, NUM_OF_BETA) if beta_max else None
    for cluster_name in CLUSTERS:
        ib_data = IB(input_matrixes[CLUSTERS_DIC[cluster_name]], subjects[CLUSTERS_DIC[cluster_name]], regions,
                     area_names, area_types, beta_values, cluster_name, normal)
        ib_data.run_analysis(analys_by)


def main():
    # main_analyze(ANALYSE_BY_AREAS)
    for i, cluster_name in enumerate(CLUSTERS):
        ib_data = load_analysed_data("data/" + SOURCE_DIR + ANALYSE_TYPE + "/" + cluster_name + "-" + ANALYSE_TYPE)
        mat, bins = Data.generate_new_prob(ib_data.input_matrix)
        plot_convergence_Dkl(ib_data)
        # plot_hierarchy(ib_data, pre_pros_for_hierarchy(ib_data))

    mat, bins = Data.generate_new_prob(np.array([[1,2,2,3, 3,3,4,5],[1,2,2,2,3,3,4, 5]]))
    breakpoint()

if __name__ == '__main__':
    main()
