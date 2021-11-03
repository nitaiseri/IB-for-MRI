from embo import InformationBottleneck
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat


class IB:
    def __init__(self, p_y_x):
        self.p_y_x = p_y_x
        self.normalized_pyx = self.normalize()

    def normalize(self):
        return self.p_y_x


def D_kl(p1, p2):
    C1 = np.einsum('ij,ik->ijk', p1, 1 / p2)  ###is it the right way?
    # print(C1)
    C2 = np.log(C1)
    # print(C2)
    C2[p1 == 0, :] = 0
    # print(C2)
    return np.einsum('ij,ijk->jk', p1, C2)


def D_kl_vec(p_y_x_hat):
    Dkl_vals = []
    for idx in range(p_y_x_hat.shape[1]):
        p_mean = np.mean(p_y_x_hat, axis=1)
        p_cur = p_y_x_hat[:, idx]
        dkl = sum([p_cur[x] * np.log(p_cur[x] / p_mean[x]) if (p_cur[x] > 0) else 0 for x in range(p_y_x_hat.shape[0])])
        # dkl = -sum([p_mean[x]*np.log(p_cur[x]/p_mean[x]) if (p_mean[x] > 0 and p_cur[x] > 0) else 0 for x in range(33)])
        Dkl_vals.append(dkl)
    return Dkl_vals


def IB_iter(p_x, p_y_x, p_x_hat_given_x, beta):
    if beta < 2800:
        # set_trace()
        pass
    p_x_hat = p_x_hat_given_x @ p_x
    p_x_given_x_hat = (p_x_hat_given_x * p_x).T / (p_x_hat)
    # p_x_given_x_hat[np.isnan(p_x_given_x_hat)] = 0
    p_y_x_hat = p_y_x @ p_x_given_x_hat
    not_norm = np.exp(-beta * D_kl(p_y_x, p_y_x_hat)) * p_x_hat
    not_norm = not_norm.T
    return not_norm / np.sum(not_norm, axis=0)


def prepare_prob(input_matrix):
    # input_matrix = abs(np.random.normal(2,1,(5,8)))
    p_y_x = input_matrix / np.sum(input_matrix, axis=0)

    x_dim = p_y_x.shape[1]
    p_x = np.ones(x_dim)
    p_x = p_x / np.sum(p_x)

    p_x_hat_given_x = np.eye(x_dim) + abs(np.random.normal(0, 0.02, (x_dim, x_dim)))
    p_x_hat_given_x = p_x_hat_given_x / np.sum(p_x_hat_given_x, axis=0)

    return p_y_x, p_x, p_x_hat_given_x


def get_clusters(input_matrix, beta_values):
    p_y_x, p_x, p_x_hat_given_x = prepare_prob(input_matrix)

    beta_values = beta_values[::-1]

    clus = []
    full_distances = []
    clusters_matrix = []
    for beta in beta_values:
        err = 1
        # while err > 1e-7:
        while err > (1 / beta) / 10:
            prev_p = p_x_hat_given_x
            p_x_hat_given_x = IB_iter(p_x, p_y_x, p_x_hat_given_x, beta)
            err = np.sum(abs(prev_p - p_x_hat_given_x))

        p_x_hat = p_x_hat_given_x @ p_x
        p_x_given_x_hat = (p_x_hat_given_x * p_x).T / (p_x_hat)

        p_y_x_hat = p_y_x @ p_x_given_x_hat
        full_distances.append(D_kl_vec(p_y_x_hat))
        # clus.append(np.linalg.matrix_rank(p_y_x_hat, tol = 1e-7))
        # print(beta, np.linalg.matrix_rank(p_y_x_hat, tol = 1e-7))
        clus.append(np.linalg.matrix_rank(p_y_x_hat, tol=(1 / beta) / 10))
        # print(beta, np.linalg.matrix_rank(p_y_x_hat, tol = (1/beta)/10))
        t, indices = np.unique(p_x_given_x_hat.round(decimals=int(np.ceil(np.log10(10 * beta)))), axis=1,
                               return_inverse=True)
        clusters_matrix.append(indices)
        # print(beta)
    # print(clusters_matrix)

    return clus, p_x_given_x_hat, p_y_x_hat, full_distances, clusters_matrix


def generate_beta(max_value = 20000, length = 800):
    beta_values = [max_value]
    for idx in range(length-1):
        beta_values.append(beta_values[-1]*0.99)
    return beta_values[::-1]


def load_data(path):
    x = loadmat(path)
    MTsat, R1, MD, R2, MTV, R2s = 0, 1, 2, 3, 4, 5
    mean_values = {}
    subjects = {}

    with open('C:\\Users\\nitai seri\\Desktop\\study\\university\\year3\\Lab\\gender.txt') as f:
         gender = f.read().splitlines()
    with open('C:\\Users\\nitai seri\\Desktop\\study\\university\\year3\\Lab\\age.txt') as f:
         age = f.read().splitlines()
    with open('C:\\Users\\nitai seri\\Desktop\\study\\university\\year3\\Lab\\area_names1.txt') as f:
         area_names = f.read().splitlines()
    with open('C:\\Users\\nitai seri\\Desktop\\study\\university\\year3\\Lab\\area_names2.txt') as f:
         area_names = f.read().splitlines()
    with open('C:\\Users\\nitai seri\\Desktop\\study\\university\\year3\\Lab\\area_types.txt') as f:
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
    print(len(area_names))
    for ix in range(6):
        print(subjects[ix].shape)
        print(mean_values[ix].shape)

    del area_types[29]  # remove Left Accumbens
    del area_types[20]  # remove Medulla
    del area_types[8]  # remove Right Accumbens
    region = {}
    for x, y in zip(area_names, area_types):
        region[x] = y


def run_analysis():
    load_data()


load_data('C:\\Users\\nitai seri\\Desktop\\study\\university\\year3\\Lab\\huji_data.mat')
