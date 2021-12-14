import pandas as pd
import numpy as np
import scipy
import nibabel as nib
from scipy import stats
from numpy import genfromtxt
import os
import glob
from re import search
import matplotlib.pyplot as plt
import seaborn as sns


class HumanScans:

    def __init__(self, Mtsat, R1, R2, R2s, Mtv, Md, seg):
        self.Mtsat = Mtsat
        self.R1 = R1
        self.R2 = R2
        self.R2s = R2s
        self.Mtv = Mtv
        self.Md = Md
        self.seg = seg


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


def load_human_data(sub_path) -> HumanScans:
    """
    # Creates HumanScan Object that have np arrays of the parameters data, t1 ,r2s ,mt, tv and seg file
    # input: sub_path: string representing the folder path of a subject to their MRI scan data
    :param sub_path: sub path of the given subject
    :return: subject object that hold his data.
    """

    # look for maps paths
    T1file = os.path.join(sub_path, 'mrQ_fixbias', 'OutPutFiles_1', 'BrainMaps', 'T1_map_Wlin.nii.gz')
    TVfile = os.path.join(sub_path, 'mrQ_fixbias', 'OutPutFiles_1', 'BrainMaps', 'TV_map.nii.gz')
    R2sfile = os.path.join(sub_path, 'multiecho_flash_R2s', 'R2_mean_2TVfixB1.nii.gz')
    MTfile = os.path.join(sub_path, 'MT', 'MT_sat_mrQ_fixbias.nii.gz')
    MDfile = os.path.join(sub_path, 'freesurfer', 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz')
    R2file = os.path.join(sub_path, 'freesurfer', 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz')
    high_seg_file = os.path.join(sub_path, 'freesurfer', 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz')
    MD_seg_file = os.path.join(sub_path, 'freesurfer', 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz')
    R2_seg_file = os.path.join(sub_path, 'freesurfer', 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz')

    if not os.path.isfile(T1file) or not os.path.isfile(TVfile) or not os.path.isfile(R2sfile) or not os.path.isfile(
            MTfile) or not os.path.isfile(segfile):
        return -1, -1  # -1 acts as a return code
    seg = nib.load(segfile)
    t1 = nib.load(T1file)
    r1 = 1 / t1.get_fdata()  # todo: problem with zero division, change later
    r1 = np.nan_to_num(r1, posinf=0.0, neginf=0.0)

    tv = nib.load(TVfile)
    tv = tv.get_fdata()


    r2s = nib.load(R2sfile)
    r2s = r2s.get_fdata()


    mt = nib.load(MTfile)
    mt = mt.get_fdata()

    seg = seg.get_fdata()

    # measures = [t1, tv, r2s, mt]  # change back to t1 once you know how to fix the division by zero
    measures = [r1, tv, r2s, mt]  # change back to r1 once you know how to fix the division by zero

    return measures, seg


def main():
    # SUBCORTICAL
    sub_cortex_dict = {10: 'Left-Thalamus-Proper', 11: 'Left-Caudate', 12: 'Left-Putamen', 13: 'Left-Pallidum',
                       17: 'Left-Hippocampus', 18: 'Left-Amygdala', 26: 'Left-Accumbens-area',
                       49: 'Right-Thalamus-Proper', 50: 'Right-Caudate', 51: 'Right-Putamen', 52: 'Right-Pallidum',
                       53: 'Right-Hippocampus', 54: 'Right-Amygdala',
                       58: 'Right-Accumbens-area'}  # TODO: GOOD VERSION DO NOT DELETE!!!!

    # corr by means
    analysisDir = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'
    subject_names = HUJI_subjects_preprocess(analysisDir)
    subject_paths, names = get_subject_paths(analysisDir, 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz',
                                             subject_names)
    mea_names = ["t1", "tv", "r2s", "mt"]
    rois = [10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54,
            58]  # subcortical regions left and right TODO: GOOD VERSION DO NOT DELETE!!!!
    mat_a = []
    mat_b = []
    for path in subject_paths:
        measures, seg = load_human_data(path)
        if measures != -1:
            mat_a.append(measures)
            mat_b.append(seg)
            return measures, seg


if __name__ == "__main__":
    main()