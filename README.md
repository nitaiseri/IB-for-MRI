# IB-for-MRI
Information Bottleneck, iterative algorithm for MRI scans.

For any questions Nitai seri : nitai.ns@gmail.com
github.com/nitaiseri/IB-for-MRI

Contains:
raw_data - directory which contain for each experiments the data 
            over subjects and areas, and probability matrices.
data - directory that contain for each experiment the data after
        was processed by the algorithm.
plots - directory that contain for each experiment the plots.
README - This file.
probability_data_maker.py - class and method to generate probability matrices 
of MRI scan.
other.py - Pieces of unused code.
load_data.py - Code to load in two options (see below) MRI scan data into 
HumanScans objects.
IB.py - The main code that run the IB algorithm and plot.
Experiments.py - Different experiments that tried with this modules and two simple examples. 
Consts.py - File of constants for all modules. 
config.py - File of global variables to update in order to run different experiments. usualy use the consts file.

part of them need to modify for each experiment (explanation below)

In order to run experiment with this code there are few steps:
0. Update the config file to match the data you want to work with and 
    the kind of experiment you want to do.
1. load the relevant data into humanScans objects.
2. create a probability tables out of it.
3. Run the iterative IB algorithm and plot.

STEPS:

## In order to see the steps on practice you can look an "huji_simple_example" experiment, 
## and "parkinson_simple_example" on the Experiments.py.

Step 1 - Load data - load_data.py:
This module load the given data into humanScans objects.
This can be done in two options:
A. 'HUJI type' data: ( As in "huji_simple_example" experiment)
   - Provide the path to the directory of the subjects scan.
     (Modify this in Consts - PATH_TO_RELEVANT_DATA_SCANS).
   - Make sure that the sub paths to all the scans and segmentation in the subjects
     are fit to those in the Consts. (Otherwise modify them)
   - Provide the path to the csv table of details of the subjects.
     (Modify this in Consts - META_DATA).
   - Make sure it is the same csv as in the example.
   - Provide the areas dictionary.
     (Modify this in Consts - AREA_DICT).
   - Modify PARAMETERS to ALL_PARAMETERS or those you want.

B.  'parkinson type' data: ( As in "parkinson_simple_example" experiment)
    - Provide the path to the directory of the subjects scan.
      (Modify this in Consts - PATH_TO_RELEVANT_DATA_SCANS).
    - Make sure that the sub paths to all the scans and segmentation in the subjects
      are fit to those in the Consts - In the parkinson part. (Otherwise modify them)
    - Provide the path to the matlab struct of details and the analysis of the subjects.
      (Modify this in Consts - ANALAYZED_DATA_DIR).
    - IMPORTANT - Make sure it is the same struct as in the example.
    - In the function load_data.load_data, also need to provide list of areas to work on.
      It is need to be compatible with the struct. (Each column is a sub area struct)
   -  Modify PARAMETERS to PARK_PARAMETERS or those you have.

In Both, supply a 'sub_path' with informative name of the specific experiment (see examples)  

Then save the data with 'load_data.save_data()'.

Step 2 - Generate probability - probability_data_maker.py:
Here only need to decide the method we want to create the probability table.
Usually I did it with all three of them. But not necessary, and I would 
recommend on the second option ('voxels_in_areas_probability').
Need to be modified in Consts - PROB_FUNC_OPTIONS. (leave it as a list).

Step 3 - Run the algorithm - IB_Nitai.py:
Here we need to decide:
A. how many beta values do we want. This is depend on the rank of the matrix.
(Which depend on how many subjects and areas do we have)
Modify in Consts - NUM_OF_BETA
B. Which axis we want to run on. 'Over areas' or 'Over people'.
Teh first make clusters of areas, and the second create clusters of subjects.
Modify in Consts - ANALYSE_BY


GENERAL NOTES:
- The may reach to numerical problems on some experiments.
  happens mainly in get_clusters function. For example when trying to run parkinson_validation_on_subjects
  with AREAS_DIC_SEPERATE.
- After running save_data , the data on the subjects and the areas saved in 'raw_data' directory
  (But not the scans or subject object. Here is the place to say that I didnt manage to save them in any
   way and this is Ba'asa because it takes time to load them)
- After Step two, The probability matrices saved in 'raw_data' directory inside sub folders.
- After step three The IB data saved in 'data' directory and the plots saved in 'plots' directory.


