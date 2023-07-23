from timeit import default_timer as timer
import warnings
import pandas as pd
from scipy.stats import kendalltau
from scipy.stats import pearsonr


class CorrelationCalculator:
    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', None)
    pd.set_option('max_seq_item', None)
    pd.set_option('display.max_rows', None)

    def __init__(self, data_root):
        self._feature_list = ["Duration", "Src_IP", "Src_Pt", "Dst_IP", "Dst_Pt", "Packets", "Bytes", "Flags"]
        self._train_filename = data_root + "training/CIDDS_Internal_train.csv"
        self._results = {}
        self._timer_start = 0
        self._timer_end = 0

    def calculate(self, func):
        print("Beginning correlation calculation")

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        self._set_execution_timer()

        df = pd.read_csv(self._train_filename)

        for i in self._feature_list:
            x = df[i]
            y = df.Label

            corr, _ = func(x, y)
            key = i
            value: float = corr
            loop_dict = {key: value}
            self._results.update(loop_dict)

        print(f"\nCalculation completed in : {self._get_execution_timer()} seconds.")

        return self._results

    # Set the current time and add to start or end
    def _set_execution_timer(self):
        if self._timer_start != 0:
            self._timer_end = timer()
        else:
            self._timer_start = timer()

    # Get total time for execution
    def _get_execution_timer(self):
        execution_time = self._timer_end - self._timer_start
        return float(f'{execution_time:.2f}')
