"""
    Corey Nolan
    MCS WPI Summer 2023
    CS534 - Artificial Intelligence
    Version4

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
"""

import time
import pandas as pd
import glob
from timeit import default_timer as timer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


class DataPrep:
    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', None)
    pd.set_option('max_seq_item', None)
    pd.set_option('display.max_rows', None)

    # **********************************************************
    def __init__(self, raw_data_path, data_root):
        #
        #   constructor
        #
        #   params:
        #       raw_data_path from main.py
        #       data_root from main.py
        #
        #
        # **********************************************************

        self._columns = ["Date first seen", "Proto", "Flows", "Tos", "attackType", "attackID", "attackDescription"]
        self._timer_start = 0
        self._timer_end = 0
        self._parse_timer = 0
        self._raw_dir = True
        self._path_truth = False
        self._raw_files = []
        self._raw_data_path = raw_data_path
        self._data_root = data_root
        self._parsed_weeks_path = "Data/datasets/CIDDS/parsed_weeks/"
        self._train_filename = self._data_root + "training/CIDDS_Internal_train.csv"
        self._test_filename = self._data_root + "testing/CIDDS_Internal_test.csv"
        self._full_dataframe = pd.DataFrame()

    # USER INPUT FOR RAW CSV OR NOT
    def set_raw_dir(self):

        # NEW RAW DATA?
        loop = True

        while loop:

            new_data = input("\nAre you starting with raw/un-parsed csv files?: (y/n) ").lower()

            if new_data == 'y' or new_data == 'yes':
                print("\n")

                list_files = []

                for files in glob.glob(self._raw_data_path + 'CIDDS-001-internal-week*'):
                    list_files.append(rf'{files}')

                if len(list_files) == 4:

                    # SET LIST OF FILES TO THE CLASS VARIABLE
                    self._raw_files = list_files
                    # EXIT THE PATH ENTRY LOOP
                    self._path_truth = True
                    # EXIT THE RAW CSV LOOP
                    loop = False
                    # SET BOOLEAN FOR RAW DIR
                    self._raw_dir = True

                else:
                    print("\n!!!!Raw Datafile Location Error!!!!\n\n"
                          "The four weekly raw data files need to be in: \n"
                          "    Data/datasets/CIDDS/CIDDS-001/\n"
                          "       CIDDS-001-internal-week1.csv\n"
                          "       CIDDS-001-internal-week2.csv\n"
                          "       CIDDS-001-internal-week3.csv\n"
                          "       CIDDS-001-internal-week4.csv\n"
                          "\n"
                          "Fix the file locations and run application again.\n")
                    time.sleep(5)
                    quit()

            elif new_data == 'n' or new_data == 'no':
                # EXIT THE RAW CSV LOOP
                loop = False
                # SET BOOLEAN FOR RAW DIR
                self._raw_dir = False

            else:
                print("\nPlease enter either yes or no")

    # GET INFO ON RAW CSV OR NOT(RETURNS T/F)
    def get_raw_dir(self):
        return self._raw_dir

    # SET THE CURRENT TIME AND ADD TO START OR END
    def _set_execution_timer(self):
        if self._timer_start != 0:
            self._timer_end = timer()
        else:
            self._timer_start = timer()

    # GET TOTAL TIME FOR EXECUTION
    def _get_execution_timer(self):
        execution_time = self._timer_end - self._timer_start
        return float(f'{(execution_time / 60):.2f}')

    # CLEAN AND NORMALIZE DATA FROM RAW CSV FILES. COMBINE INTO ONE DATAFRAME
    def set_parse_data(self):

        self._set_execution_timer()

        counter = 1

        for i in self._raw_files:
            print(f"Parsing data from Week{counter} of 4...")
            df = pd.read_csv(i, low_memory=False)

            # DROP COLUMNS
            print("Dropping unnecessary columns..")
            df.drop(self._columns, axis=1, inplace=True)

            # RENAME COLUMNS TO REMOVE SPACES FOR EASIER HANDLING
            print("Renaming columns...")
            df = df.rename(
                columns={'class': 'Label', 'Src IP Addr': 'Src_IP', 'Src Pt': 'Src_Pt', 'Dst IP Addr': 'Dst_IP',
                         'Dst Pt': 'Dst_Pt', })

            # DROP ROWS WITH EMPTY VALUES(IF ANY)
            df.dropna(axis=0, inplace=True, how="any")

            print("Normalizing the data...")

            # CONVERT DURATION AND SRC_PT TO INT64
            df['Duration'] = df['Duration'].astype('int64')
            df['Dst_Pt'] = df['Dst_Pt'].astype('int64')

            # DROP SRC_IP ROWS THAT CONTAIN ANYCAST ADDRESSES (0.0.0.0)
            df = df[df.Src_IP != '0.0.0.0']

            # DROP SRC_IP ROWS THAT CONTAIN BROADCAST ADDRESSES (255.255.255.255)
            df = df[df.Src_IP != '255.255.255.255']

            # DROP DST_IP ROWS THAT CONTAIN ANYCAST ADDRESSES (0.0.0.0)
            df = df[df.Dst_IP != '0.0.0.0']

            # DROP DST_IP ROWS THAT CONTAIN BROADCAST ADDRESSES (255.255.255.255)
            df = df[df.Dst_IP != '255.255.255.255']

            #  REPLACE SRC_IP INSTANCES OF 192 IN SRC_IP WITH 0
            df.loc[df['Src_IP'].str.contains('192.'), 'Src_IP'] = '0'

            # REPLACE SRC_IP INSTANCES OF RANDOMLY GENERATED NUMBERS WITH 1
            df.loc[df['Src_IP'].str.contains('1'), 'Src_IP'] = '1'

            # REPLACE SRC_IP INSTANCES OF DNS WITH 1
            df.loc[df['Src_IP'].str.contains('DNS'), 'Src_IP'] = '1'

            # REPLACE SRC_IP INSTANCES OF EXT WITH 1
            df.loc[df['Src_IP'].str.contains('EXT'), 'Src_IP'] = '1'

            #  REPLACE DST_IP INSTANCES OF 192 IN SRC_IP WITH 0
            df.loc[df['Dst_IP'].str.contains('192.'), 'Dst_IP'] = '0'

            # REPLACE DST_IP INSTANCES OF RANDOMLY GENERATED NUMBERS WITH 1
            df.loc[df['Dst_IP'].str.contains('1'), 'Dst_IP'] = '1'

            # REPLACE DST_IP INSTANCES OF DNS WITH 1
            df.loc[df['Dst_IP'].str.contains('DNS'), 'Dst_IP'] = '1'

            # REPLACE DST_IP INSTANCES OF EXT WITH 1
            df.loc[df['Dst_IP'].str.contains('EXT'), 'Dst_IP'] = '1'

            #  REPLACE INSTANCES OF NORMAL IN CLASS WITH 0
            df.loc[df['Label'].str.contains('normal'), 'Label'] = '0'

            # REPLACE INSTANCES OF ATTACKER WITH 1
            df.loc[df['Label'].str.contains('attacker'), 'Label'] = '1'

            # REPLACE INSTANCES OF VICTIM WITH 1
            df.loc[df['Label'].str.contains('victim'), 'Label'] = '1'

            # CREATE A LIST TO STORE TCP VALUES
            flags_list = []

            # ITERATE THROUGH THE ROWS OF THE FLAGS COLUMN
            for j in df.Flags:
                # CALL THE CONVERT FUNCTION ON EACH ROW VALUE
                value = self._convert_tcp(j)
                # APPEND NEW VALUE TO LIST
                flags_list.append(value)

            # OVERWRITE THE STR VALUES IN THE FLAGS COLUMN WITH
            # NEW INT VALUES
            df['Flags'] = flags_list

            # CREATE A LIST FOR THE NEW VALUES
            bytes_list = []
            # FOR EACH VALUE IN THE BYTES ROW
            for k in df.Bytes:
                # IF M IS IN THE STRING
                if "M" in str(k):
                    # CALL THE BYTES CALC FUNCTION
                    bytes_list.append(self._calculate_bytes(k))
                else:
                    bytes_list.append(k)

            # OVERWRITE THE CURRENT BYTES COLUMN WITH NEW VALUES
            df['Bytes'] = bytes_list

            self._full_dataframe = pd.concat([self._full_dataframe, df], axis=0)

            print(f"Done with week{counter}.\n")

            counter += 1

        # CALCULATE TIME AND PRINT TO SCREEN
        self._set_execution_timer()
        self._parse_timer = self._get_execution_timer()
        print(f"Parsing completed in : {self._parse_timer} minutes.")

    # DEDUPLICATE DATA, SPLIT TO TRAIN/TEST, RANDOM UNDER SAMPLE TEST DATA
    # OUTPUT THE TRAINING AND TESTING CSV
    def split_data(self):

        self._set_execution_timer()

        print("\nSplitting data into Train(70%) and Test(30%)...")

        print(f"\nCount of datapoints before deduplication: {len(self._full_dataframe)}")

        # DROP ALL DUPLICATE ROWS FROM THE DATAFRAME
        self._full_dataframe.drop_duplicates(keep='first', inplace=True)
        print(f"Count of datapoints after deduplication: {len(self._full_dataframe)}")

        # SPLIT THE DATAFRAME INTO TRAINING AND TESTING, SHUFFLE TO RANDOMIZE
        train, test = train_test_split(self._full_dataframe, test_size=.30, shuffle=True)

        print(f"Number of points in the split test set : {len(test)}")
        print(f"Number of points in the split train set: {len(train)}")

        print("\nCount of normal labels in the training set")
        print(train['Label'].value_counts()[0])

        print("\nCount of abnormal labels in the training set")
        print(train['Label'].value_counts()[1])

        print("\nCount of normal labels in the testing set before RandomUnderSampling:")
        print(test['Label'].value_counts()[0])

        print("\nCount of abnormal labels in the testing set before RandomUnderSampling:")
        print(test['Label'].value_counts()[1])

        # RESAMPLE TESTING DATA IN ORDER TO EVEN OUT NORMAL/ABNORMAL
        features = test.drop('Label', axis=1)
        target = test['Label']
        sample = RandomUnderSampler(sampling_strategy="not minority")
        features_sample, target_sample = sample.fit_resample(features, target)
        test = features_sample.join(target_sample)
        print(f'\nFull number of testing datapoints after RandomUnderSampling: \n{len(test)}')

        print("\nCount of normal labels in the testing set after RandomUnderSampling")
        print(test['Label'].value_counts()[0])

        print("\nCount of abnormal labels in the testing set after RandomUnderSampling")
        print(test['Label'].value_counts()[1])

        # EXPORT TRAINING AND TESTING FILE TO RESPECTIVE FOLDERS
        train.to_csv(self._train_filename, index=False)
        test.to_csv(self._test_filename, index=False)

        # PRINT FILE LOCATION TO SCREEN
        print(f"\nTraining and Testing files can be found at:\n"
              f"    {self._train_filename}\n"
              f"    {self._test_filename}")

        # SET STOP TIMER
        self._set_execution_timer()
        complete_time = self._get_execution_timer()
        split_timer = complete_time - self._parse_timer

        # CALCULATE TIME AND PRINT TO SCREEN
        print(f"\nSplitting data completed in : {split_timer} minutes.")
        print(f"Data Prep completed in : {complete_time} minutes.\n")

    # CONVERT MEGA BYTES TO BYTES
    @staticmethod
    def _calculate_bytes(mega):
        data_bytes = int(float(mega.split()[0]) * 1000000)
        return data_bytes

    # CONVERT CATEGORICAL TCP FLAGS TO A SUM OF INT NMBRS
    @staticmethod
    def _convert_tcp(string):
        count = 0
        for i in string:

            if i == "A":
                count += 1
            elif i == "P":
                count += 2
            elif i == "R":
                count += 3
            elif i == "S":
                count += 4
            elif i == "F":
                count += 5
        return count
