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
    + test_SVM.py: test file to verify class implementation of SVM outside of UI
    + test_DT.py: test file to verify class implementation of DT outside of UI

Known issues when running main.py:
    + not all model classes are implemented yet 
    + UI not complete
        - add an escape (like 'q') to quit early
        - loop main so we can test more than one model/thing at a time

TODO's: 
    + complete UI in main.py
    + implement the SVM render
    + probably should only create one subfolder under the model folder with the 
    model name, in this folder there should be a trained model file, a results file,
    and a .png of the model render
        - this will require changes to Utilities.utilities get/set file locations
        - verify that we have covered all error checking necessary
        - name the model <model_name> + "_model.pkl"
    + add coefficient correlation script for selecting SVM columns
    + implement warm_start on models so we can do incremental training

