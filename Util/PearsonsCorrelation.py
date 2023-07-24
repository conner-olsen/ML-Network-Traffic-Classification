'''
    CN
    CS534 - Artificial Intelligence

    Calculates the Pearsons R correlation coefficient between the target and all features.

    Required paramater: DATA_ROOT

    Uses the CIDDS_Internal_train.csv located at:
        "Data/datasets/CIDDS/training/CIDDS_Internal_train.csv"

    Returns a dictionary of Features and their correlation to the Target.
        {'Duration': -0.04397088966420225, 'Src_IP': -0.14384103803592363,
        'Src_Pt': 0.013573661322596221, 'Dst_IP': -0.12330950654573887,
         'Dst_Pt': -0.017577922630247426, 'Packets': -0.04354099928161065,
          'Bytes': -0.0930107277338663, 'Flags': 0.10467687857110357}

'''


import pandas as pd
from scipy.stats import pearsonr
from timeit import default_timer as timer

class PearsonsCorrelation:

    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', None)
    pd.set_option('max_seq_item', None)
    pd.set_option('display.max_rows', None)

    # **********************************************************
    def __init__(self, data_root):
        #
        #   constructor
        #
        #   params:
        #       data_root from main.py
        #
        #
        # **********************************************************

        self._feature_list = ["Duration", "Src_IP", "Src_Pt", "Dst_IP", "Dst_Pt", "Packets", "Bytes", "Flags"]
        self._train_filename = data_root+"training/CIDDS_Internal_train.csv"
        self._pearson_results = {}
        self._timer_start = 0
        self._timer_end = 0


    def set_pearson_calc(self):

        print("Beginning Pearsons R Calc...")

        self._set_execution_timer()

        df = pd.read_csv(self._train_filename)

        for i in self._feature_list:

            x = df[i]
            y = df['Label']

            corr, _ = pearsonr(x, y)
            key = i
            value: float = corr
            loop_dict = {key: value}
            self._pearson_results.update(loop_dict)

        self._set_execution_timer()

        print(f"\nPearsons R completed in : {self._get_execution_timer()} seconds.")

        return self._pearson_results

    ## SET THE CURRENT TIME AND ADD TO START OR END
    def _set_execution_timer(self):
        if self._timer_start != 0:
            self._timer_end = timer()
        else:
            self._timer_start = timer()

    ## GET TOTAL TIME FOR EXECUTION
    def _get_execution_timer(self):
        execution_time = self._timer_end - self._timer_start
        return float(f'{execution_time:.2f}')
        # return float(f'{(execution_time / 60):.2f}')


