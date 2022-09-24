#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:13:02 2022

@author: nitai.seri
"""
import numpy as np
import random
import os
import load_data
from Consts import *
from config import *
import IB
import probability_data_maker as pdm


# More Information about the first two example experiments is in the README file.

def huji_simple_example():
    """
    This is a simple experiment that created as an example how to run an experiment - With HUJI data type.
    First Modify the relevant Consts:
    * RUN = HUJI_TYPE -> Because The type of raw data we work on is the first type as in the README.
    * PATH_TO_RELEVANT_DATA_SCANS = DATA_DIR_HUJI
    Where DATA_DIR_HUJI = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'
    Which is the path to the subjects scan.
    * META_DATA = HUJI_META_DATA
    Where HUJI_META_DATA = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human/meta_ed.csv'
    Which is the data over the subjects.
    * AREA_DICT = SUB_CORTEX_DICT -> Dictionary of the areas. You can see examples in Consts.
    * PROB_FUNC_OPTIONS - List of the option to make the probability function.
    * NUM_OF_BETA
    * ANALYSE_BY = ANALYSE_BY_AREAS -> Because we want to clustre by areas for example.
    :return: None
    """
    # This is the name of the experiment. (For the directories names and graphs)
    sub_path = "/huji_simple_example/"
    raw_path = RAW_DATA_PATH + sub_path
    # Step one:
    subjects = load_data.load_data()
    load_data.save_data(subjects, sub_path)
    # Step two:
    pdm.generate_data(subjects, sub_path)
    # Step three
    IB.main(raw_path[:-1], sub_path)

def parkinson_simple_example():
    """
    This is a simple experiment that created as an example how to run an experiment - With parkinson data type.
    First Modify the relevant Consts:
    * RUN = PARK_TYPE -> Because The type of raw data we work on is the first type as in the README.
    * PATH_TO_RELEVANT_DATA_SCANS = DATA_DIR_PARK
    Where DATA_DIR_PARK = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Parkinson_SZ'
    Which is the path to the subjects scan.
    * ANALAYZED_DATA_DIR = ELIOR_ANALYZED_DATA_DIR
    Where ELIOR_ANALYZED_DATA_DIR = r'/ems/elsc-labs/mezer-a/nitai.seri/Desktop/IB-for-MRI/raw_data/Elior_meta_data' + R1_ANALYZED_DATA
    Which is the data and analysis over the subjects.
    * Choose on which areas and group wand to run.
      This case "right_caudate" and PARKINSON_GROUP.
    * PROB_FUNC_OPTIONS - List of the option to make the probability function.
    * NUM_OF_BETA
    * ANALYSE_BY = ANALYSE_BY_AREAS -> Because we want to clustre by areas for example.
    :return: None
    """
    # This is the name of the experiment. (For the directories names and graphs)
    sub_path = "/park_simple_example/"
    raw_path = RAW_DATA_PATH + sub_path
    # Step one:
    subjects = load_data.load_data(AREAS_DIC_SEPERATE["right_caudate"], PARKINSON_GROUP)
    load_data.save_data(subjects, sub_path)
    # Step two:
    pdm.generate_data(subjects, sub_path)
    # Step three
    IB.main(raw_path[:-1], sub_path)


def two_halfs_partition(subjects):
    """
    Function to devide subjects into two equal size homogeneous parts (by young-old and male-female)
    :param subjects: list of subjects
    :return: two lists of subjects
    """
    bad_partiton = True
    while (bad_partiton):
        bad_partiton = False
        np.random.shuffle(subjects)
        subs1 = subjects[:len(subjects)//2]
        subs2 = subjects[len(subjects)//2:]
        if abs(len([sub.gender for sub in subs1 if sub.gender == 'F']) -
               len([sub.gender for sub in subs2 if sub.gender == 'F'])) > 2:
            bad_partiton = True
        if abs(len([sub.age for sub in subs1 if sub.age > 50]) -
               len([sub.age for sub in subs2 if sub.age > 50])) > 2:
            bad_partiton = True   
    return subs1, subs2


def basic_validation():
    """
    Simple function that helps to make sure that the algorithm/data is stable.
    By separate the subjects into two homogeneous groups, and expecting to get similar
    clustering hierarchy.
    :return: None
    """
    sub_path = "/basic_validation_homogen"
    subjects = load_data.load_data()
    subs1, subs2 = two_halfs_partition(subjects)
    subs1_path = sub_path + "/first/"
    subs2_path = sub_path + "/second/"
    load_data.save_data(subs1, subs1_path)
    load_data.save_data(subs2, subs2_path)
    pdm.generate_data(subs1, subs1_path)
    pdm.generate_data(subs2, subs2_path)
    IB.main(RAW_DATA_PATH + subs1_path[:-1], subs1_path)
    IB.main(RAW_DATA_PATH + subs2_path[:-1], subs2_path)

        
def huji_bootstrap():
    """
    Simple function that helps to make sure that the algorithm/data is stable.
    By bootstraping 80% from the subjects, and expecting to get similar clustering hierarchy.
    :return: None
    """
    sub_path = "/bootstrap_validation"
    subjects = load_data.load_data()
    precent = 0.8
    for i in range(10):
        subs = random.sample(subjects, k=round(len(subjects) * precent))
        subs_path = RAW_DATA_PATH + sub_path + "/" + str(i) + "/"
        load_data.save_data(subs, sub_path + "/")
        pdm.generate_data(subs, sub_path + "/")
        IB.main(subs_path[:-1], sub_path + "/" + str(i) + "/")

        
def parkinson_bootstrap():
    """
    Simple function that helps to make sure that the algorithm/data is stable.
    By bootstraping 80% from the subjects, and expecting to get similar clustering hierarchy.
    For each group (parkinson and control), run over sub area and combining areas.
    :return: None
    """
    for group in GROUPS.keys():
        for areas_dic in AREAS_DIC:
            for area in areas_dic:
                sub_path = "/" + group + "/" + area
                subjects = load_data.load_data(areas_dic[area], GROUPS[group])
                precent = 0.8
                for i in range(10):
                    subs = random.sample(subjects, k=round(len(subjects) * precent))
                    subs_path = RAW_DATA_PATH + sub_path + "/" + str(i) + "/"
                    load_data.save_data(subs, sub_path + "/")
                    pdm.generate_data(subs, sub_path + "/")
                    IB.main(subs_path[:-1], sub_path + "/" + str(i) + "/", group + "-" + area + "-" + str(i))
        

def parkinson_validation():
    """
    Function that run the algorithm over parkinson subjects and control subjects,
    and cluster by areas in order to see if there are sub areas that indicate more
    about the existence of the disease. (Need to modify constants)
    :return: None
    """
    for areas_dic in AREAS_DIC:
        for area in areas_dic:
            sub_path = "/all_without_caudate_7/" + area
            subjects = load_data.load_data(areas_dic[area], PARKINSON_GROUP)
            subjects += load_data.load_data(areas_dic[area], CONTROL_GROUP)
            load_data.save_data(subjects, sub_path + "/")
            pdm.generate_data(subjects, sub_path + "/")
            IB.main(RAW_DATA_PATH + sub_path, sub_path + "/", "all_without_caudate_7" + "-" + area)

def parkinson_validation_on_subjects():
    """
    Function that run the algorithm over parkinson subjects and control subjects,
    and cluster by subjects in order to see if it can recognise and classify the
    two groups. (Need to modify constants)
    :return: None
    """
    #for areas_dic in AREAS_DIC:
    areas_dic = AREAS_DIC_ALL
    for area in areas_dic:
        sub_path = "/" + "subjects_colors" + "/" + area
        subjects = load_data.load_data(areas_dic[area], PARKINSON_GROUP)
        subjects += load_data.load_data(areas_dic[area], CONTROL_GROUP)
        load_data.save_data(subjects,sub_path + "/")
        pdm.generate_data(subjects,sub_path + "/")
        IB.main(RAW_DATA_PATH + sub_path, sub_path + "/", "subjects" + "-" + area)
    
if __name__ == '__main__':
    parkinson_validation()
