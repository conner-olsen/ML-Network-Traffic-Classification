# CS534_FinalProject
Network Traffic Classification project using Artificial Intelligence

Data parsing/cleaning/splitting for the CIDDS-001 data sets containing 31,287,933 datapoints total

Cleans weekly internal traffic datasets. Splits into Testing and Training set at 70/30.

Requires Input Raw Data Files to be in proper location:
    /Data/datasets/CIDDS/CIDDS-001/
        CIDDS-001-internal-week1.csv
        CIDDS-001-internal-week2.csv
        CIDDS-001-internal-week3.csv
        CIDDS-001-internal-week4.csv
 
Provides Training and Testing csv files as output to:

Data/datasets/CIDDS/training/CIDDS_Internal_train.csv
    -roughly 7,390,904 data points
    - 7,254,919 Label = 0 (Normal)
    - 136,021   Label = 1 (Abnormal)
        
Data/datasets/CIDDS/testing/CIDDS_Internal_test.csv    
    - roughly 116,972 datapoints
    - 58486     Label = 0 (Normal)
    - 58486     Label = 0 (Abnormal)       