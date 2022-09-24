#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:05:30 2022

@author: nitai.seri
"""

import probability_data_maker as pdm

####################################################################
################### probability_data_maker.py ######################
####################################################################

### Used to plot the histogram of the probability gaussian.
### Shouldn't be change for the experiment.
PLOT = False


# three different methods to generate the probability data.
# Even if choose one, leave it in a list.
ALL_PROB_FUNC_OPTIONS = [
    (pdm.ProbabilityMaker.mean_probability_data, "per_mean"),
    (pdm.ProbabilityMaker.voxels_in_areas_probability, "per_area"),
    (pdm.ProbabilityMaker.total_voxels_probability, "total")
]

####################################################################
######################## IB_Nitai.py ##############################
####################################################################

CLUSTERS_DIC = {"MTsat": 0, "R1": 1, "MD": 2, "R2": 3, "MTV": 4, "R2s": 5}
MTsat, R1, MD, R2, MTV, R2s = 0, 1, 2, 3, 4, 5
ANALYSE_BY_AREAS = True
ANALYSE_BY_PEOPLE = False
ANALIZE_STRINGS = {ANALYSE_BY_AREAS: "ANALYSE_BY_AREAS",
                   ANALYSE_BY_PEOPLE: "ANALYSE_BY_PEOPLE"}
COLOR_LABELS = False

####################################################################
######################## load_data.py ##############################
####################################################################

SUB_CORTEX_DICT = {10: 'Left-Thalamus-Proper', 11: 'Left-Caudate', 12: 'Left-Putamen', 13: 'Left-Pallidum',
                   17: 'Left-Hippocampus', 18: 'Left-Amygdala', 26: 'Left-Accumbens-area',
                   49: 'Right-Thalamus-Proper', 50: 'Right-Caudate', 51: 'Right-Putamen', 52: 'Right-Pallidum',
                   53: 'Right-Hippocampus', 54: 'Right-Amygdala',
                   58: 'Right-Accumbens-area'}  # GOOD VERSION DO NOT DELETE!!!!

AREA_DICT_TRY = {10: 'Left-Thalamus-Proper', 11: 'Left-Caudate', 12: 'Left-Putamen', 13: 'Left-Pallidum',
                   17: 'Left-Hippocampus', 18: 'Left-Amygdala', 26: 'Left-Accumbens-area',
                   49: 'Right-Thalamus-Proper', 50: 'Right-Caudate', 51: 'Right-Putamen', 52: 'Right-Pallidum',
                   53: 'Right-Hippocampus', 54: 'Right-Amygdala',58: 'Right-Accumbens-area',2:'Left-Cerebellum-White-Matter',
                   3:'Left-Cerebellum-Cortex',41:'Right-Cerebellum-White-Matter',47:'Right-Cerebellum-Cortex'}
### segmentation maps names ###
HIGH_SEG = "high_seg"
MD_SEG = "MD_seg"
R2_SEG = "R2_seg"

### scans type names ###
MT_SAT = "Mtsat"
R1 = "R1"
R2 = "R2"
R2S = "R2s"
MTV = "Mtv"
MD = "Md"

ALL_PARAMETERS = [MT_SAT, R1, R2, R2S, MTV, MD]
PARK_PARAMETERS = [R1, R2S, MTV]

### segmentation type for each scan ###
SEG_MAP = {MT_SAT: HIGH_SEG,
           R1: HIGH_SEG,
           R2: R2_SEG,
           R2S: HIGH_SEG,
           MTV: HIGH_SEG,
           MD: MD_SEG}

#################################

#############
### Paths ###
#############

MAIN_PATH = "/ems/elsc-labs/mezer-a/nitai.seri/Desktop/IB-for-MRI/"

RAW_DATA_PATH = MAIN_PATH + "raw_data"

DATA_DIR_HUJI = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'
DATA_DIR_PARK = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Parkinson_SZ'

### HUJI ###
# Sub paths to scans
SUB_PATH_HUJI_T1 = 'mrQ_fixbias/OutPutFiles_1/BrainMaps/T1_map_Wlin.nii.gz'
SUB_PATH_HUJI_MTV = 'mrQ_fixbias/OutPutFiles_1/BrainMaps/TV_map.nii.gz'
SUB_PATH_HUJI_R2s = 'multiecho_flash_R2s/R2_mean_2TVfixB1.nii.gz'
SUB_PATH_HUJI_MTsat = 'MT/MT_sat_mrQ_fixbias.nii.gz'
SUB_PATH_HUJI_MD = 'Dif_fsl_preprocessed/eddy/aligned2T1/dtiInit/dti94trilin/bin/MD_2mrQ.nii.gz'
SUB_PATH_HUJI_T2 = 'T2/R2map.nii.gz'
SUB_PATH_HUJI_HIGH_SEG = 'freesurfer/segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz'
SUB_PATH_HUJI_MD_SEG = 'Dif_fsl_preprocessed/eddy/aligned2T1/dtiInit/dti94trilin/bin/segFSLMPRAGE_BS_wmparc_B1corrALL_2DTI_resamp.nii.gz'
SUB_PATH_HUJI_R2_SEG = 'T2/segFSLMPRAGE_BS_wmparc_B1corrALL_2T2_resamp_BM.nii.gz'

HUJI_META_DATA = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human/meta_ed.csv'

### Parkinson ###
SUB_PATH_PARK_R1 = 'mrQ_2022/OutPutFiles_1/BrainMaps/R1_map.nii.gz'
SUB_PATH_PARK_MTV = 'mrQ_2022/OutPutFiles_1/BrainMaps/TV_map.nii.gz'
SUB_PATH_PARK_R2s = 'multiecho_flash_R2s/R2_mean_2TV.nii.gz'

R1_ANALYZED_DATA = '/RG_PD_SZ_R1_median.mat'
MTV_ANALYZED_DATA = '/RG_PD_SZ_MTV_median.mat'
R2s_ANALYZED_DATA = '/RG_PD_SZ_R2*_median.mat'

ELIOR_ANALYZED_DATA_DIR = r'/ems/elsc-labs/mezer-a/nitai.seri/Desktop/IB-for-MRI/raw_data/Elior_meta_data' + R1_ANALYZED_DATA
#################################


### Parkinson analysis consts ###

# Rows in the matlab struct
PARKINSON_GROUP = 0
CONTROL_GROUP = 1

# Columns in the matlab struct
LEFT_CAUDATE = 0
RIGHT_CAUDATE = 1
LEFT_PUTAMAN = 2
RIGHT_PUTAMAN = 3

GROUPS = {"parkinson" : PARKINSON_GROUP, "control" : CONTROL_GROUP}
AREAS = [LEFT_CAUDATE, RIGHT_CAUDATE, LEFT_PUTAMAN, RIGHT_PUTAMAN]

AREAS_DIC_SEPERATE = {"right_caudate" : [RIGHT_CAUDATE], "left_caudate" : [LEFT_CAUDATE], "left_putamen": [LEFT_PUTAMAN], "right_putamen" : [RIGHT_PUTAMAN]}
AREAS_DIC_TWO = {"caudate" : [LEFT_CAUDATE, RIGHT_CAUDATE], "putamen": [LEFT_PUTAMAN, RIGHT_PUTAMAN]}
AREAS_DIC_ALL = {"caudate&putamen" : [LEFT_CAUDATE, RIGHT_CAUDATE, LEFT_PUTAMAN, RIGHT_PUTAMAN]}
AREAS_DIC = [AREAS_DIC_SEPERATE, AREAS_DIC_TWO, AREAS_DIC_ALL]


############################
HUJI_TYPE = 0
PARK_TYPE = 1
