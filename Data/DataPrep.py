"""
    Corey Nolan, Conner Olsen
    MCS WPI Summer 2023
    CS534 - Artificial Intelligence
    Version4

    Data parsing/cleaning/splitting for the CIDDS-001 data sets containing 31,287,933 datapoints totals.

    Cleans weekly internal traffic datasets.
    Splits into Testing and Training set at 70/30.
"""

import os
import glob
import time
import pandas as pd
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


def _convert_tcp(string):
    """
    Converts categorical TCP flags to a sum of integers.

    :param string: TCP flags string.
    :return: Sum of the integer representations TCP flags.
    """
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


def _calculate_bytes(mega):
    """
    Converts MegaBytes to Bytes.

    :param mega: Value in MegaBytes.
    :return: Value in Bytes.
    """
    calculated_bytes = int(float(mega.split()[0]) * 1000000)
    return calculated_bytes


class DataPrep:
    """Class for data preparation."""

    pd.set_option("display.max_columns", None)
    pd.set_option("max_colwidth", None)
    pd.set_option("max_seq_item", None)
    pd.set_option("display.max_rows", None)

    def __init__(self, raw_data_path, data_root, convert_strings=True):
        """
        Initializes the class with file paths and settings.

        :param raw_data_path: Path to raw data.
        :param data_root: Root directory for data.
        :param convert_strings: Flag to convert strings to numeric, default is True.
        """
        self._columns = [
            "Date first seen",
            "Proto",
            "Flows",
            "Tos",
            "attackType",
            "attackID",
            "attackDescription",
        ]
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
        self._train_filename_resamp = (
            self._data_root + "training/CIDDS_Internal_train_resample.csv"
        )
        self._test_filename_resamp = (
            self._data_root + "testing/CIDDS_Internal_test_resample.csv"
        )
        self._full_dataframe = pd.DataFrame()
        self._convert_strings = convert_strings  # default turn strings into numeric

    def set_raw_dir(self):
        """
        Set raw directory for files.
        """
        list_files = []

        for files in glob.glob(self._raw_data_path + "CIDDS-001-internal-week*"):
            list_files.append(rf"{files}")

        if len(list_files) == 4:
            self._raw_files = list_files
            self._path_truth = True
            self._raw_dir = True
        else:
            print(
                "\n!!!!Raw Datafile Location Error!!!!\n\n"
                "The four weekly raw data files need to be in: \n"
                "    Data/datasets/CIDDS/CIDDS-001/\n"
                "       CIDDS-001-internal-week1.csv\n"
                "       CIDDS-001-internal-week2.csv\n"
                "       CIDDS-001-internal-week3.csv\n"
                "       CIDDS-001-internal-week4.csv\n"
                "\n"
                "Fix the file locations and run application again.\n"
            )
            time.sleep(5)
            quit()

    def get_raw_dir(self):
        """
        Returns the raw directory.

        :return: Raw directory.
        """
        return self._raw_dir

    def _set_execution_timer(self):
        """
        Sets the execution timer.
        """
        if self._timer_start != 0:
            self._timer_end = timer()
        else:
            self._timer_start = timer()

    def _get_execution_timer(self):
        """
        Gets the execution timer.

        :return: Execution time.
        """
        execution_time = self._timer_end - self._timer_start
        return float(f"{(execution_time / 60):.2f}")

    def set_parse_data(self, convert_strings):
        """
        Parses data from raw files.

        :param convert_strings: Flag to convert strings to numeric.
        """
        self._set_execution_timer()

        counter = 1

        for i in self._raw_files:
            print(f"\nParsing data from Week {counter} of 4...")
            df = pd.read_csv(i, low_memory=False)

            print("Dropping unnecessary columns..")
            df.drop(self._columns, axis=1, inplace=True)

            print("Renaming columns...")
            df = df.rename(
                columns={
                    "class": "Label",
                    "Src IP Addr": "Src_IP",
                    "Src Pt": "Src_Pt",
                    "Dst IP Addr": "Dst_IP",
                    "Dst Pt": "Dst_Pt",
                }
            )

            df.dropna(axis=0, inplace=True, how="any")

            print("Normalizing the data...")
            df["Duration"] = df["Duration"].astype("int64")
            df["Dst_Pt"] = df["Dst_Pt"].astype("int64")

            df = df[df.Src_IP != "0.0.0.0"]
            df = df[df.Src_IP != "255.255.255.255"]
            df = df[df.Dst_IP != "0.0.0.0"]
            df = df[df.Dst_IP != "255.255.255.255"]

            ip_replace_dict = {"192.": "0", "1": "1", "DNS": "1", "EXT": "1"}
            for old_value, new_value in ip_replace_dict.items():
                df.loc[df["Src_IP"].str.contains(old_value), "Src_IP"] = new_value
                df.loc[df["Dst_IP"].str.contains(old_value), "Dst_IP"] = new_value

            label_replace_dict = {"normal": "0", "attacker": "1", "victim": "1"}
            for old_value, new_value in label_replace_dict.items():
                df.loc[df["Label"].str.contains(old_value), "Label"] = new_value

            flags_list = []
            if convert_strings:
                for j in df.Flags:
                    flags_list.append(_convert_tcp(j))
                df["Flags"] = flags_list

            bytes_list = []
            for k in df.Bytes:
                if "M" in str(k):
                    bytes_list.append(_calculate_bytes(k))
                else:
                    bytes_list.append(k)
            df["Bytes"] = bytes_list

            self._full_dataframe = pd.concat([self._full_dataframe, df], axis=0)

            print(f"Done with week{counter}.\n")

            counter += 1

        self._set_execution_timer()
        self._parse_timer = self._get_execution_timer()
        print(f"Parsing completed in : {self._parse_timer} minutes.")

    def split_data(self, resample=True):
        """
        Splits the data into training and testing sets.

        :param resample: Flag to resample the dataset, default is True.
        """
        self._set_execution_timer()

        print("\nSplitting data into Train(70%) and Test(30%)...")

        print(
            f"\nCount of datapoints before deduplication: {len(self._full_dataframe)}"
        )

        self._full_dataframe.drop_duplicates(inplace=True)
        print(f"Count of datapoints after deduplication: {len(self._full_dataframe)}")

        train, test = train_test_split(self._full_dataframe, test_size=0.30)

        print(f"Number of points in the split test set : {len(test)}")
        print(f"Number of points in the split train set: {len(train)}")

        if not os.path.exists(self._data_root + "training/"):
            os.makedirs(self._data_root + "training/")
        if not os.path.exists(self._data_root + "testing/"):
            os.makedirs(self._data_root + "testing/")

        train.to_csv(self._train_filename, index=False)
        test.to_csv(self._test_filename, index=False)

        if not resample:
            if not self._convert_strings:
                self._train_filename = self._train_filename.replace(
                    ".csv", "_strings.csv"
                )
                self._test_filename = self._test_filename.replace(
                    ".csv", "_strings.csv"
                )
            print(
                f"\nNon-resampled training and testing files can be found at:\n"
                f"    {self._train_filename}\n"
                f"    {self._test_filename}"
            )
        else:
            print("\nCount of normal labels in the training set")
            print(train["Label"].value_counts()[0])

            print("\nCount of abnormal labels in the training set")
            print(train["Label"].value_counts()[1])

            print(
                "\nCount of normal labels in the testing set before RandomUnderSampling:"
            )
            print(test["Label"].value_counts()[0])

            print(
                "\nCount of abnormal labels in the testing set before RandomUnderSampling:"
            )
            print(test["Label"].value_counts()[1])

            features = test.drop("Label", axis=1)
            target = test["Label"]
            sample = RandomUnderSampler(sampling_strategy="not minority")
            features_sample, target_sample = sample.fit_resample(features, target)
            test = features_sample.join(target_sample)

            print(
                f"\nFull number of testing datapoints after RandomUnderSampling: \n{len(test)}"
            )

            print(
                "\nCount of normal labels in the testing set after RandomUnderSampling"
            )
            print(test["Label"].value_counts()[0])

            print(
                "\nCount of abnormal labels in the testing set after RandomUnderSampling"
            )
            print(test["Label"].value_counts()[1])

            train.to_csv(self._train_filename_resamp, index=False)
            test.to_csv(self._test_filename_resamp, index=False)

            print(
                f"\nResampled training and testing files can be found at:\n"
                f"    {self._train_filename_resamp}\n"
                f"    {self._test_filename_resamp}"
            )

        self._set_execution_timer()
        complete_time = self._get_execution_timer()
        split_timer = complete_time - self._parse_timer

        print(f"\nSplitting data completed in : {split_timer: .2f} minutes.")
        print(f"Data Prep completed in : {complete_time: .2f} minutes.\n")
