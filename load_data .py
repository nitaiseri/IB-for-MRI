### Imports ###

import pandas as pd
import numpy as np
import pickle
import scipy
import nibabel as nib
from typing import Union
from scipy import stats
from scipy import io
import os
from re import search
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import csv

from Consts import *
from config import *
###################


class HumanScans:
    """
    class that represent and keep data of MRI scans for one subject.
    """

    def __init__(self, Mtsat=None, R_1=None, R_2=None, R2s=None, Mtv=None
                 , Md=None, high_seg=None, MD_seg=None, R2_seg=None
                 , name=None, age=None, gender=None, group_type=None):
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
        self.areas_dic = None
        self.group_type = group_type
        self.segmentation = {HIGH_SEG: high_seg,
                             MD_SEG: MD_seg,
                             R2_SEG: R2_seg}
        self.parameters = {MT_SAT: Mtsat,
                           R1: R_1,
                           R2: R_2,
                           R2S: R2s,
                           MTV: Mtv,
                           MD: Md}
        
    def set_area_dic(self, area_dic):
        """
        Setter func to set the area dictionary of the HumanScans object.
        :param area_dic: the dictionary area. key - segmentation number in free-surfer, value - name of area.
        :return: None
        """
        self.areas_dic = area_dic

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
        for area in self.areas_dic.keys():
            nums = self.get_indexes_of_area(parameter, area).T
            if nums.shape[0] == 0:
                print(parameter, " ", self.areas_dic[area], "\n")
            values_per_area.append(self.parameters[parameter][nums[0], nums[1], nums[2]])
        return values_per_area

    def get_mean_std_per_param(self, parameter):
        """
        calculate the mean and std value per area in the subject
        :param parameter: the type of scan
        :return: np array of array of the means per area and array of the std per area.
        """
        return np.array([(np.mean(area_vals), np.std(area_vals)) for area_vals in self.get_values_voxel(parameter)]).T

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
        voxels = voxels[voxels > 0]
        # This Std calculation get from Aviv recommendation.
        stds = np.ones(voxels.shape[0])*np.mean(voxels) * 0.03
        return voxels, stds

    def get_all_voxels(self, parameter):
        """
        find all the values of the scan in the brain
        :param parameter: the type of scan
        :return: np array of all voxels in the scans
        """
        return self.parameters[parameter][np.nonzero(self.segmentation[SEG_MAP[parameter]])]

def subjects_directory_preprocess(analysisDir, clean = False):
    """
    # Description: preprocesses subjects directory. filter irrelevant subjects.
    :param analysisDir: A path to the dir from which the data will be taken
    :param clean: Boolean. true if all the sub directories in 'analysisDir' is
                  valid.
    :return: np array shape: (num_of_subjects after preprocess, 1)
    """
    subject_names = np.array(os.listdir(analysisDir))

    ## HUJI_subjects_preprocess ##
    if not clean:
        ### HUJI diroectory clean up
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
        subject_names.remove('H062_SM')
        subject_names = np.array(subject_names)

    subject_names = subject_names.reshape(-1, 1)
    return subject_names

def get_subject_paths(analysisDir, clean_directory = False):
    """
    this func finds the paths to the MRI scan data of the subjects and creates a np array of them
    as well as a np array of the names of the subjects which have a path, returning both arrays
    analysisDir: the directory of the dataset used
    will create an array of the sub_path of each subject
    IMPORTANT - In each human directory, if the is not only one date, need to be
    'readme' file that contain the relevant date.
    :param analysisDir: path for the directory of the data scans.
    :param clean_directory : IMPORTENT - clean = True if all sub folders in 
    'analysisDir' is relevant. otherwise False and need to edit the condition 
    code in 'HUJI_subjects_preprocess' in order to filter as you wish.
    :return: list of relevant scans directories paths ans list of its names.
    """
    subject_names = subjects_directory_preprocess(analysisDir, clean_directory)
    subject_paths = []

    for sub in range(len(subject_names)):
        subject = subject_names[sub][0]
        os.chdir(analysisDir)
        scanedate = os.path.join(analysisDir, subject)  # adds the to the string the second string so we have a path
        os.chdir(scanedate)  # goes to current path
        readmepath = os.path.relpath('readme', scanedate)  # This method returns a string value which represents the relative file path to given path from the start directory.
        if os.path.isfile(readmepath):  # has a read me
            file1 = open(readmepath, 'r')
            A = file1.readlines()[0].rstrip("\n")
            sub_path = scanedate + '/' + A
        else:
            subfolders = glob.glob(scanedate + '/*/')
            if subfolders == []:
                continue
            sub_path = subfolders[0]
        subject_paths.append(sub_path)
    return subject_paths, subject_names

def load_table(path, r=False):
    """
    Load a nifty file into numpy table.
    :param path: path to the nifty file.
    :param r: True if it's R1 or R2 scans. (Not quite sure why).
    :return: A numpy table from the file.
    """
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
    T1file = os.path.join(sub_path, SUB_PATH_HUJI_T1)
    MTVfile = os.path.join(sub_path, SUB_PATH_HUJI_MTV)
    R2sfile = os.path.join(sub_path, SUB_PATH_HUJI_R2s)
    MTsatfile = os.path.join(sub_path, SUB_PATH_HUJI_MTsat)
    MDfile = os.path.join(sub_path, SUB_PATH_HUJI_MD)
    T2file = os.path.join(sub_path, SUB_PATH_HUJI_T2)
    high_seg_file = os.path.join(sub_path, SUB_PATH_HUJI_HIGH_SEG)
    MD_seg_file = os.path.join(sub_path, SUB_PATH_HUJI_MD_SEG)
    R2_seg_file = os.path.join(sub_path, SUB_PATH_HUJI_R2_SEG)

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

def load_all_HUJI_subjects():
    """
    This function load all HUJI type data into subjects.
    :return: list of HumanScans objects.
    """
    subject_paths, subject_names = get_subject_paths(PATH_TO_RELEVANT_DATA_SCANS)
    subjects = []
    subject_details = pd.read_csv(META_DATA)
    names = subject_details["SubID"].values
    ages = subject_details["Age"].values
    genders = subject_details["Sex"].values
    additional_data = list(zip(names, ages, genders))
    for i, path in enumerate(subject_paths):
        sub = load_human_data(path, additional_data[i])
        if sub:
            subjects.append(sub)
    relevant_area_dic = AREA_DICT

    for sub in subjects:
        sub.set_area_dic(relevant_area_dic)
    return subjects

def save_data(subjects, sub_path):
    """
    Save as numpy table information about the subjects and the areas for late in the IB functions.
    :param subjects: list of the subjects.
    :param path: path to where to save those tables.
    :return: None
    """
    path = RAW_DATA_PATH + sub_path
    subjects_data = []
    if not os.path.exists(path):
            os.makedirs(path)
    for i, subject in enumerate(subjects):
        subjects_data.append([i, subject.age, subject.gender, subject.name, subject.group_type])
    with open(path + 'subjects.npy', 'wb') as f:
                np.save(f, subjects_data)
    with open(path + 'areas.npy', 'wb') as f:
                np.save(f, list(subjects[0].areas_dic.values()))

def load_pk_data(sub_path, additional_data):
    """
    Init incomplete HumanScan object.
    :param sub_path: path to the relevant subject scans.
    :param additional_data: ID, sex, age.
    :return: HumanScan object.
    """
    R1file = os.path.join(sub_path, SUB_PATH_PARK_R1)
    R1 = load_table(R1file, r=True)
    
    MTVfile = os.path.join(sub_path, SUB_PATH_PARK_MTV)
    Mtv = load_table(MTVfile)
    
    R2sfile = os.path.join(sub_path, SUB_PATH_PARK_R2s)
    R2s = load_table(R2sfile)
    
    name_, age_, gender_, group_type_ = additional_data
    return HumanScans(R_1=R1, R2s=R2s, Mtv=Mtv, name=name_, age=age_, gender=gender_, group_type=group_type_)

def create_pk_subjects(sub_rg, group_type):
    """
    Initial all the HumanScan objects (without segmentation and more)
    :param sub_rg: The Data of the first column of the specific group.
    (just for the data above all the subjects - Does'nt supposed to be different between columns)
    :param group_type: The group (raw) we are working on.
    :return: List of uncompleted HumanScan objects
    """

    subjects_names = sub_rg['subject_names'].tolist()
    
    subjects_ages = sub_rg['age'].tolist()
    
    subjects_sex = sub_rg['sex'].tolist()
    
    types = [group_type] * len(subjects_names)
    
    additional_data = list(zip(subjects_names, subjects_ages, subjects_sex, types))[:]
    
    subjects_paths = [(PATH_TO_RELEVANT_DATA_SCANS if subname.startswith('PD') else DATA_DIR_HUJI)
                      + "/" + subname for subname in subjects_names]
    
    subjects = []

    for i, path in enumerate(subjects_paths):
        sub = load_pk_data(path, additional_data[i])
        if sub:
            subjects.append(sub)
    
    return subjects

def update_subjects_segmentation(subjects, individual_data, seg_counter, slice_num):
    """
    This function create the segmentation map according to the data in the analyzed data path (the matlab file).
    :param subjects: List of HumanScan objects  that need ot be update.
    :param individual_data: The struct that contain the indicies of the sub areas.
    :param seg_counter: the number of the segmentation of the subarea
    :param slice_num: number of the slice.
    :return: None
    """
    for i, subject in enumerate(subjects):
        indicies = individual_data[i]['segment_inds'][0][slice_num]
        if subject.segmentation[HIGH_SEG] is None:
            subject.segmentation[HIGH_SEG] = subject.parameters[R1]*0
        for x, y, z in indicies:
            subject.segmentation[HIGH_SEG][x-1][y-1][z-1] = seg_counter

def load_pk_subjects(areas, group):
    """
    This function load the second type of data into HumanScans object.
    :param areas: list of areas that needed to be count as areas of brain in this experiment.
                    in the second type of raw data, this is number of column in the matlab struct.
    :param group: the group that we are experiment above.
                in the second type of raw data, this is number of raw in the matlab struct.
    :return: list of HumanScans objects.
    """
    rg = scipy.io.loadmat (ANALAYZED_DATA_DIR, squeeze_me=True)['RG']
    
    subjects = create_pk_subjects(rg[group][0], group)
            
    areas_dictionary = {}
    
    seg_counter = 1
    
    for area in areas:
        sub_rg = rg[group][area]

        area_name = sub_rg['ROI_label'].tolist()
        
        group_name = sub_rg['group_name'].tolist()
        
        individual_data = sub_rg['individual_data'].tolist()
        
        for slice_num in range(SLICES):
            # condition to ignore caudate 7
            # if (area == 0 or area == 1) and (slice_num == 6):
            #     continue
            update_subjects_segmentation(subjects, individual_data, seg_counter, slice_num)
            areas_dictionary[seg_counter] = area_name + "-" +  str(seg_counter % 7 if (seg_counter % 7)!=0 else 7)
            seg_counter += 1
        
    for sub in subjects:
         sub.set_area_dic(areas_dictionary)            
    
    return subjects
    
def load_data(areas=None ,group=None):
    """
    Main function of this class. create list oj HumanScans objects. And save information over subjects and areas.
    :param areas: list of areas - for the second option.
    :param group: group number - for the second option.
    :return: list of HumanScans objects.
    """
    if RUN:
        subjects = load_pk_subjects(areas, group)
    else:
        subjects = load_all_HUJI_subjects()
    return subjects

if __name__ == "__main__":
    load_data()

