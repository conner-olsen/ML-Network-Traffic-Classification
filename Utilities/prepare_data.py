'''
    author: em
    project: Network Traffic Classification AI
    class: CS 534 Artificial Intelligence
    school: WPI
    date: June 30, 2023
    last update: July 6, 2023

    function for preparing data for training/testing
    NOTE: most of this was moved to the class files, retaining for debug
'''
import pandas as pd
import time

#*******************************************************************
@staticmethod
def remove_extra_columns(filename, drop_cols):
#   
#   function to remove specified array of columns
#   
#   params:
#       filename: csv file to open
#       drop_cols: string array, name of column(s) to drop from csv
#   returns:
#       nothing, print function
#*******************************************************************
    # Load the data
    df = pd.read_csv(filename, low_memory=False)
    print(df.columns)

    #Drop the unnecessary columns
    for c in drop_cols:
        df = df.drop(columns=[c])

    # Write the data back to a new CSV file
    df.to_csv(filename, index=False)
    print("Done")
    time.sleep(5000)



#**********************************************************************
@staticmethod
def load_and_prepare_data(model_type, filename, attributes):
#
#   Load the from the given file and prepare it for 
#       supervised training or testing.
#       Note: main program flow is separating load and process
#     prepares the data for training (DT)
#       convert 'bytes' col to numeric, set null vals to 0
#       convert all categorical data to integers
#       drop label column from x (data)
#       set y = label column (target)
#
#   params:
#       filename (str): the name of the file to load (the dataset)
#       attributes (str array): the attribute names (column headers)
#
#   Returns:
#       x, y (2 tuples): preprocessed feature matrix and label vector
#       x: samples aka data
#       y: targets aka labels
#***********************************************************************    
    # Load the dataset
    df = pd.read_csv(filename, low_memory=False)

    # Convert 'Bytes' column to numeric, setting non-numeric values to NaN
    df['Bytes'] = pd.to_numeric(df['Bytes'], errors='coerce')
    df['Bytes'] = df['Bytes'].fillna(0)  # Fill NaN values with 0

    # Factorize categorical features to integer labels
    for col in attributes:
        df[col], _ = pd.factorize(df[col])

    x = df.drop(columns=['Label'])
    y = df['Label']

    #debug print
    print(str(len(df)) + " examples in dataset")

    return x, y

#*********************************************************
@staticmethod
def load_dataset(filename):
#
#   read dataset into DataFrame structure
#   
#   params:
#       filename (str): name of the file to load
#   returns:
#       DataFrame structure (pandas) of dataset
#*********************************************************
    df =  pd.read_csv(filename, low_memory=False)
    df.info(verbose=True)   #print details about the dataset
    return df