# CS534_FinalProject
Network Traffic Classification project using Artificial Intelligence

Project is organized as such, under the root folder (CS534_FinalProject):
    + Data folder
        - contains dataset .csv files 
        - contains scripts for processing the data into the training and testing files
    + DT, FKM, SVM, TACGAN folders
        - one folder for each model
        - class file for model: each class has IDENTICAL user function names for running from main agnostically, though implementation may vary according to the specific model type
            * any model-specific helper functions in addition to the user functions are considered private and to be accessed within that class file only
        - subfolder for stored models
        - subfolder for test results
        - subfolder for renders 
    + Utilities
        - files in this folder have shared functions for all models
    
The main project root folder contains:
    + main.py: user interface, main project driver file