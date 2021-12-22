import pandas as pd
import numpy as np
import pickle
import scipy
import nibabel as nib
from typing import Union
import json
from scipy import stats
from numpy import genfromtxt
import os
import glob
from re import search
import matplotlib.pyplot as plt
import seaborn as sns

# SUBCORTICAL
SUB_CORTEX_DICT = {10: 'Left-Thalamus-Proper', 11: 'Left-Caudate', 12: 'Left-Putamen', 13: 'Left-Pallidum',
                   17: 'Left-Hippocampus', 18: 'Left-Amygdala', 26: 'Left-Accumbens-area',
                   49: 'Right-Thalamus-Proper', 50: 'Right-Caudate', 51: 'Right-Putamen', 52: 'Right-Pallidum',
                   53: 'Right-Hippocampus', 54: 'Right-Amygdala',
                   58: 'Right-Accumbens-area'}  # TODO: GOOD VERSION DO NOT DELETE!!!!

HIGH_SEG = "high_seg"
MD_SEG = "MD_seg"
R2_SEG = "R2_seg"
MT_SAT = "Mtsat"
R1 = "R1"
R2 = "R2"
R2S = "R2s"
MTV = "Mtv"
MD = "Md"
PARAMETERS = [MT_SAT, R1, R2, R2S, MTV, MD]
SEG_MAP = {MT_SAT: HIGH_SEG,
           R1: HIGH_SEG,
           R2: R2_SEG,
           R2S: HIGH_SEG,
           MTV: HIGH_SEG,
           MD: MD_SEG}


class HumanScans:
    """
    class that represent and keep data of MRI scans for one subject.
    """

    def __init__(self, Mtsat, R_1, R_2, R2s, Mtv, Md, high_seg, MD_seg, R2_seg, name, age, gender):
        """
        initial one human subject.
        :param Mtsat: Mtsat scan
        :param R_1: R_1 scan
        :param R_2: R_2 scan
        :param R2s: R2s scan
        :param Mtv: Mtv scan
        :param Md: Md scan
        :param high_seg: segmentation map for all others
        :param MD_seg: segmentation map for MD
        :param R2_seg: segmentation map for R2
        :param name: name of subject
        :param age: age of subject
        :param gender: gender of subject
        """
        self.name = name
        self.gender = gender
        self.age = age
        self.segmentation = {HIGH_SEG: high_seg,
                             MD_SEG: MD_seg,
                             R2_SEG: R2_seg}
        self.parameters = {MT_SAT: Mtsat,
                           R1: R_1,
                           R2: R_2,
                           R2S: R2s,
                           MTV: Mtv,
                           MD: Md}

    def get_indexes_of_area(self, parameter, area_num):
        """
        return all the indices of specific area of the subject
        :param parameter: the type of scan
        :param area_num: number of area to find(according to freeSurfer)
        :return: nd array of all the indices of the area
        """
        return np.argwhere(self.segmentation[SEG_MAP[parameter]] == area_num)

    def get_num_of_voxels_in_areas(self,parameter, area_nums):
        """
        return number of voxels in specific area.
        :param parameter: the type of scan
        :param area_nums: number of area to find(according to freeSurfer)
        :return: number of voxels in specific area.
        """
        return self.get_indexes_of_area(parameter, area_nums).shape[0]

    def get_values_voxel(self, parameter):
        """
        find all the values of voxels area.
        :param parameter: the type of scan
        :return: list of nd array of the voxels values per area
        """
        values_per_area = []
        for area in SUB_CORTEX_DICT.keys():
            nums = self.get_indexes_of_area(parameter, area).T
            values_per_area.append(self.parameters[parameter][nums[0], nums[1], nums[2]])
        return values_per_area

    def get_mean_per_param(self, parameter):
        """
        calculate the mean value per area in the subject
        :param parameter: the type of scan
        :return: np array of the means per area.
        """
        return np.array([np.mean(area_vals) for area_vals in self.get_values_voxel(parameter)])

    def get_all_voxels_per_areas(self, parameter):
        """
        find all the values of the scan in the brain
        :param parameter: the type of scan
        :return: np array of the voxels in all relevant area.
        """
        voxels = np.array([])
        arrs = self.get_values_voxel(parameter)
        for arr in arrs:
            voxels = np.concatenate((voxels, arr))
        return voxels

    def get_all_voxels(self, parameter):
        """
        find all the values of the scan in the brain
        :param parameter: the type of scan
        :return: np array of all voxels in the scans
        """
        return self.parameters[parameter][np.nonzero(self.parameters[parameter])]


def HUJI_subjects_preprocess(analysisDir):
    """
    # Description: preprocesses HUJI subjects by the consensus in the lab
    :param analysisDir: A path to the dir from which the data will be taken
    :return: np array shape: (num_of_subjects after preprocess, 1)
    """

    subject_names = os.listdir(analysisDir)
    subject_names = [i for i in subject_names if (search('H\d\d_[A-Z][A-Z]', i) or search('H\d\d\d_[A-Z][A-Z]', i))]
    # currently this regex is good for the 'H010_AG' and 'H60_GG' expressions
    subject_names.sort()
    del subject_names[1:8]

    subject_names.remove('H010_AG')
    subject_names.remove('H011_GP')
    subject_names.remove('H014_ZW')
    subject_names.remove('H029_ON')
    subject_names.remove('H057_YP')
    subject_names.remove('H60_GG')
    subject_names.remove('H061_SE')
    subject_names = np.array(subject_names)

    subject_names = subject_names.reshape(-1, 1)
    return subject_names


def get_subject_paths(analysisDir, niba_file_name, subject_names):
    """
    # this func finds the paths to the MRI scan data of the subjects and creates a np array of them
    # as well as a np array of the names of the subjects which have a path, returning both arrays
    # analysisDir: the directory of the dataset used
    # niba_file_name (nii.gz end ing)
    # will create an array of the sub_path of each subject
    # ex1: analysisDir: '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'
    # niba_file_name: 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz'
    # subject_names: numpy array of names of the subjects: shape:(num of subjects, 1), vector, return value from
    # HUJI_subjects_preprocess
    # ex2: analysisDir: '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'
    # niba_file_name: 'first_all_none_firstseg.nii.gz'
    :param analysisDir:
    :param niba_file_name:
    :param subject_names:
    :return:
    """

    subject_paths = []
    names = []
    for sub in range(len(subject_names)):
        subject = subject_names[sub]
        os.chdir(analysisDir)
        scanedate = os.path.join(analysisDir, subject[0])  # adds the to the string the second string so we have a path
        os.chdir(scanedate)  # goes to current path
        readmepath = os.path.relpath('readme', scanedate)  # This method returns a string value which represents the relative file path to given path from the start directory.
        if os.path.isfile(readmepath):  # has a read me
            file1 = open(readmepath, 'r')
            A = file1.readlines()[0]
            sub_path = scanedate + '/' + A

        else:
            subfolders = glob.glob(scanedate + '/*/')
            if subfolders == []:
                # print(subject, "no folder")
                continue

            sub_path = subfolders[0]
        subject_paths.append(sub_path)
        names.append(subject[0])
    return subject_paths, subject_names


def load_table(path, r=False):
    table = nib.load(path).get_fdata()
    if r:
        table = 1 / table  # todo: problem with zero division, change later
        table = np.nan_to_num(table, posinf=0.0, neginf=0.0)
    return table


def load_human_data(sub_path, additional_data) -> Union[HumanScans, None]:
    """
    # Creates HumanScan Object that have np arrays of the parameters data, t1 ,r2s ,mt, tv and seg file
    # input: sub_path: string representing the folder path of a subject to their MRI scan data
    :param sub_path: sub path of the given subject
    :return: subject object that hold his data.
    """

    # look for maps paths
    T1file = os.path.join(sub_path, 'mrQ_fixbias', 'OutPutFiles_1', 'BrainMaps', 'T1_map_Wlin.nii.gz')
    MTVfile = os.path.join(sub_path, 'mrQ_fixbias', 'OutPutFiles_1', 'BrainMaps', 'TV_map.nii.gz')
    R2sfile = os.path.join(sub_path, 'multiecho_flash_R2s', 'R2_mean_2TVfixB1.nii.gz')
    MTsatfile = os.path.join(sub_path, 'MT', 'MT_sat_mrQ_fixbias.nii.gz')
    MDfile = os.path.join(sub_path, 'Dif_fsl_preprocessed', 'eddy', 'aligned2T1', 'dtiInit', 'dti94trilin', 'bin', 'MD_2mrQ.nii.gz')
    T2file = os.path.join(sub_path, 'T2', 'R2map.nii.gz')
    high_seg_file = os.path.join(sub_path, 'freesurfer', 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz')
    MD_seg_file = os.path.join(sub_path, 'Dif_fsl_preprocessed', 'eddy', 'aligned2T1', 'dtiInit', 'dti94trilin', 'bin', 'segFSLMPRAGE_BS_wmparc_B1corrALL_2DTI_resamp.nii.gz')
    R2_seg_file = os.path.join(sub_path, 'T2', 'segFSLMPRAGE_BS_wmparc_B1corrALL_2T2_resamp_BM.nii.gz')

    if not (os.path.isfile(T1file) and os.path.isfile(MTVfile) and os.path.isfile(R2sfile) and os.path.isfile(
            MTsatfile) and os.path.isfile(MDfile) and os.path.isfile(T2file) and os.path.isfile(high_seg_file) and
            os.path.isfile(MD_seg_file) and os.path.isfile(R2_seg_file)):
        return None

    high_seg = load_table(high_seg_file)
    Mtsat = load_table(MTsatfile)
    R1 = load_table(T1file, r=True)
    R2 = load_table(T2file, r=True)
    R2s = load_table(R2sfile)
    Mtv = load_table(MTVfile)
    Md = load_table(MDfile)
    MD_seg = load_table(MD_seg_file)
    R2_seg = load_table(R2_seg_file)
    name, age, gender = additional_data

    return HumanScans(Mtsat, R1, R2, R2s, Mtv, Md, high_seg, MD_seg, R2_seg, name, age, gender)


def load_all_subjects():
    analysisDir = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'
    subject_names = HUJI_subjects_preprocess(analysisDir)
    subject_paths, names = get_subject_paths(analysisDir, 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz',
                                             subject_names)
    subjects = []
    subject_details = pd.read_csv('/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human/meta_ed.csv')
    names = subject_details["SubID"].values
    ages = subject_details["Age"].values
    genders = subject_details["Sex"].values
    additional_data = list(zip(names, ages, genders))[:-1]
    for i, path in enumerate(subject_paths):
        sub = load_human_data(path, additional_data[i])
        if sub:
            subjects.append(sub)
            return sub
    return subjects

    # with open("subjects_clean_raw_data", 'wb') as subjects_scans:
    #     pickle.dump(tuple(subjects), subjects_scans)
    # with open('subjects_clean_raw_data', 'rb') as subjects_scans:
    #     s = pickle.load(subjects_scans)
    # return s

def main():
    # subcortical regions left and right TODO: GOOD VERSION DO NOT DELETE!!!!
    rois = [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]

    subject = load_all_subjects()


    for parameter in PARAMETERS:
        mean = subject.get_mean_per_param(parameter)
    # summm = np.sum(np.array(nums))

    # with open("subjects_clean_raw_data", 'rb') as ib_data:
    #     sub = pickle.load(ib_data)

if __name__ == "__main__":
    # with open('subjects_clean_raw_data', 'rb') as f:
    #     subjects = pickle.load(f)
    # data = [load_all_subjects()]
    # with open('sub', "wb") as f:
    #     pickle.dump(len(data), f)
    #     for value in data:
    #         pickle.dump(value, f)
    data2 = []
    with open('sub', "rb") as f:
        for _ in range(pickle.load(f)):
            data2.append(pickle.load(f))
    a=1