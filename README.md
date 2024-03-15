# Final Project of CS534 - Artificial Intelligence
Network Traffic Classification project using Artificial Intelligence.

The project is organized as follows under the root folder:
-  Data folder
  - Contains dataset .csv files
  - Contains scripts for processing the data into the training and testing files
-  DT, FKM, SVM, TACGAN folders
  - One folder for each model
  - Class file for each model: each class has identical user function names for running from the main agnostically, though implementation may vary according to the specific model type
    - Any model-specific helper functions in addition to the user functions are considered private and should be accessed within that class file only
  - Subfolder for stored models
  - Subfolder for test results
  - Subfolder for renders
-  Utilities
  - Files in this folder have shared functions for all models

The main project root folder contains:
-  main.py: user interface, main project driver file
