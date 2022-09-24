from Consts import *

############################
### Modify before run!!! ###
############################

# This consts are used in the models of the experiments. 
# In order to run different experiments config this global variable.
# You may the constants in the Consts.py


# STEP 1 #
RUN = PARK_TYPE
PATH_TO_RELEVANT_DATA_SCANS = DATA_DIR_PARK
PARAMETERS = PARK_PARAMETERS
# HUJI TYPE:
# Path to simple table contain data of subID, age, sex (order by subID)
# See example in '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human/meta_ed.csv'
META_DATA = HUJI_META_DATA
# Dictionary of the relevant areas in the brain. Keys- the segmentation number
# as in the freeSurfer segmentation. Values - the area name. (Always can be added as needed)
AREA_DICT = SUB_CORTEX_DICT
# OR
# PARK Type:
ANALAYZED_DATA_DIR = ELIOR_ANALYZED_DATA_DIR
# Num of slices in each sub area:
SLICES = 7

# STEP 2 #

PROB_FUNC_OPTIONS = ALL_PROB_FUNC_OPTIONS

# STEP 3 #
ANALYSE_BY = ANALYSE_BY_AREAS
NUM_OF_BETA = 700

# No need to change
ANALYSE_TYPE = ANALIZE_STRINGS[ANALYSE_BY]