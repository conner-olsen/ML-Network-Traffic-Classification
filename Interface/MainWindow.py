"""
    author: co
    project: group 7 - term project
    class: CS-534 Artificial Intelligence WPI
    date: July 15, 2023,
    last update: July 25, 2023

    main file to control program flow
"""
import datetime
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QPushButton,
    QLabel,
    QComboBox,
    QStackedWidget,
    QLineEdit,
    QApplication,
)
from Data.DataPrep import DataPrep
from Models.DT import DT
from Models.SVM import SVM
from Models.KM import KM
from Models.FKM import FKM
from Util.Util import (
    get_results_location,
    get_model_location,
    load_dataset,
)

DATA_ROOT = "Data/datasets/CIDDS/"
TRAIN = "training/CIDDS_Internal_train"
TEST = "testing/CIDDS_Internal_test"
raw_data_path = "Data/datasets/CIDDS/CIDDS-001/"

strings_train_filename = DATA_ROOT + TRAIN + "_strings.csv"
strings_test_filename = DATA_ROOT + TEST + "_strings.csv"

whole_train_filename = DATA_ROOT + TRAIN + ".csv"
whole_test_filename = DATA_ROOT + TEST + ".csv"

resample_train_filename = DATA_ROOT + TRAIN + "_resample.csv"
resample_test_filename = DATA_ROOT + TEST + "_resample.csv"

strings_resample_train_filename = DATA_ROOT + TRAIN + "_resample_strings.csv"
strings_resample_test_filename = DATA_ROOT + TEST + "_resample_strings.csv"


MODEL_NAMES = ["DT", "SVM", "KM", "FKM"]
opts = ["TRAIN", "K-FOLD TRAIN & VALIDATE", "TEST", "PROCESS DATA", "QUIT"]
parallel = joblib.Parallel(n_jobs=-1, prefer="threads")
resample = True
convert_str = False


def k_fold_xy(x, y, idx, size):
    """
    Segment the k subsets of the data for k-folds training and validation.
    Note: this is done in development on a training set.

    :param x: The prepared data to be partitioned.
    :param y: The prepared labels to be partitioned.
    :param idx: K-val * iteration.
    :param size: Size of each k-subset.
    :return: Divided data/labels for train and validate cycle (&evaluate).
    """
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    front_x = x.iloc[:idx, :]  # start at 0-test subset idx
    back_x = x.iloc[idx + size :, :]  # start after a set
    front_y = y.iloc[:idx]
    back_y = y.iloc[idx + size :]

    frames_x = [front_x, back_x]  # put together front and back
    frames_y = [front_y, back_y]
    x_train = pd.concat(frames_x)
    y_train = pd.concat(frames_y)

    x_test = x.iloc[idx : idx + size, :]  # prepare a test section
    y_test = y.iloc[idx : idx + size]

    return x_train, y_train, x_test, y_test


# **************************************************************
def test(x_test, y_test, model_type, model_obj):
    """
    Test the model after training.

    :param x_test: Test data.
    :param y_test: Test labels.
    :param model_type: Type of model being trained.
    :param model_obj: Model object.
    :return: None. Saves evaluation data.
    """
    model_location = get_model_location(
        models_folder=model_obj.model_type, model_name=model_obj.model_name
    )
    if model_location is not None and os.path.exists(model_location):
        trained_model = load_saved_model(model_type, model_obj.model_name)
        model_obj.model = trained_model
    predictions = model_obj.test_model(x_test)
    model_obj.evaluate(y_test, predictions)


# ****************************************************************
def load_saved_model(model_type, model_name):
    """
    Load a saved model if it exists.

    :param model_type: Type of model.
    :param model_name: Name of a model.
    :return: Saved model or None if error.
    """
    try:
        model = joblib.load(get_model_location(model_type, model_name))
        print("Loading model...")
    except FileNotFoundError:
        model = None
        print("could not load model")

    return model


# noinspection PyUnresolvedReferences
class MainWindow(QMainWindow):
    """
    Main window for the application.

    :param QMainWindow: Main window.
    :return: None
    """

    def __init__(self):
        """
        Initialize the main window.

        :return: None
        """
        super().__init__()
        self.complete_label = None
        self.is_done = False
        self.timer = None
        self.start_time = None
        self.thread = None
        self.threads = []
        self.status_label = QLabel()
        self.model_type = None
        self.resample = True
        self.convert_str = False
        self.model_obj = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.length = None
        self.selected_file_label = None
        self.file_dialog = None
        self.output_textbox = None
        self.setWindowTitle("CS534 Final Project")

        self.stacked_widget = QStackedWidget()  # Stacked widget

        self.create_main_menu()
        self.create_training_page()
        self.create_complete_page()

        # Set the stacked widget as the central widget
        self.setCentralWidget(self.stacked_widget)

    def create_main_menu(self):
        """
        Create the main menu page.
        :return: None
        """
        layout = QVBoxLayout()  # Main menu layout

        # ComboBox
        self.model_type = QComboBox()
        self.model_type.addItems(MODEL_NAMES)
        layout.addWidget(QLabel("Select the model type"))
        layout.addWidget(self.model_type)

        # File Dialog
        self.file_dialog = QPushButton("Select Training Data")
        self.file_dialog.clicked.connect(self.open_file_dialog)
        self.selected_file_label = QLabel("No file selected")
        layout.addWidget(self.file_dialog)
        layout.addWidget(self.selected_file_label)

        self.selected_file_label.setText(whole_train_filename)

        # Create a new QLineEdit widget
        self.output_textbox = QLineEdit()
        self.output_textbox.setText("model_name")

        # Add the textbox to your layout
        layout.addWidget(self.output_textbox)

        # Training, Validating, Testing buttons
        hbox = QHBoxLayout()
        prepare_button = QPushButton("Prepare")
        train_button = QPushButton("Training")
        validate_button = QPushButton("K-Fold Validation")
        test_button = QPushButton("Testing")

        # Connect button click signals to the appropriate
        prepare_button.clicked.connect(self.prepare_raw_data)
        train_button.clicked.connect(self.go_to_training_page)
        validate_button.clicked.connect(self.go_to_validating_page)
        test_button.clicked.connect(self.go_to_testing_page)

        hbox.addWidget(prepare_button)
        hbox.addWidget(train_button)
        hbox.addWidget(validate_button)
        hbox.addWidget(test_button)

        layout.addLayout(hbox)

        widget = QWidget()  # Wrap layout in a QWidget
        widget.setLayout(layout)

        self.stacked_widget.addWidget(widget)  # Add widget to the stack

    def create_training_page(self):
        """
        Create the training page.`
        :return: None
        """
        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        render_button = QPushButton("Render")
        render_button.clicked.connect(self.render_trained)
        layout.addWidget(render_button)
        back_button = QPushButton("Back to Main Menu")
        back_button.clicked.connect(self.go_to_main_menu)
        layout.addWidget(back_button)

        widget = QWidget()
        widget.setLayout(layout)

        self.stacked_widget.addWidget(widget)

    def create_complete_page(self):
        """
        Create the complete page.
        :return: None
        """
        layout = QVBoxLayout()
        self.complete_label = QLabel("Complete")
        layout.addWidget(self.complete_label)
        back_button = QPushButton("Back to Main Menu")
        back_button.clicked.connect(self.go_to_main_menu)
        layout.addWidget(back_button)

        widget = QWidget()
        widget.setLayout(layout)

        self.stacked_widget.addWidget(widget)

    def render_trained(self):
        """
        Render the trained model.
        :return: None
        """
        self.model_obj.render_model(self.x_train, self.y_train)

    def go_to_main_menu(self):
        """
        Go to the main menu.
        :return: None
        """
        self.stacked_widget.setCurrentIndex(0)

    def make_model(self):
        """
        Make the model object based on the selected model type
        :return: None
        """
        self.length = -1
        model_type = self.model_type.currentText()

        if model_type == "DT":
            self.model_obj = DT(self.output_textbox.text())
        elif model_type == "SVM":
            self.model_obj = SVM(model_name=self.output_textbox.text())
        elif model_type == "FKM":
            self.model_obj = FKM(model_name=self.output_textbox.text())
        elif model_type == "KM":
            self.model_obj = KM(model_name=self.output_textbox.text())
        else:
            print("invalid option, exiting application")

    def prepare_raw_data(self):
        """
        Prepare the raw data.
        :return: None
        """
        self.stacked_widget.setCurrentIndex(2)
        self.is_done = False
        self.make_model()

        # Start the timer
        self.start_time = datetime.datetime.now()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)  # update every second

        self.thread = WorkerThread(lambda: self.run_prepare())
        self.thread.task_done.connect(self.on_task_done)
        self.thread.start()
        time.sleep(0.1)

    def run_prepare(self):
        """
        Run the prepare function in a new thread.
        :return: None
        """
        # BEGIN CLEANING/NORMALIZATION/TRAIN AND TEST SPLIT OF RAW DATA

        # INSTANTIATE THE DATAPREP CLASS
        data_opt = DataPrep(raw_data_path, DATA_ROOT)

        # IF USER HAS A RAW CSV TO PARSE
        data_opt.set_raw_dir()

        if data_opt.get_raw_dir():
            data_opt.set_parse_data(self.convert_str)
            data_opt.split_data(self.resample)

        self.is_done = True
        self.update_timer()  # update one last time when training is done

    def go_to_validating_page(self):
        """
        Go to the validating page.
        :return: None
        """
        self.is_done = False
        self.stacked_widget.setCurrentIndex(2)
        self.make_model()

        df = load_dataset(resample_test_filename)
        self.x_test, self.y_test = self.model_obj.prepare_data(df, self.length)

        # Start the timer
        self.start_time = datetime.datetime.now()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)  # update every second

        # Start the validation process
        self.validate()

    def go_to_testing_page(self):
        """
        Go to the complete page for testing.
        :return: None
        """
        self.is_done = False
        self.stacked_widget.setCurrentIndex(2)

        self.make_model()

        if self.resample:
            df = load_dataset(resample_test_filename)
        else:
            df = load_dataset(whole_test_filename)
        self.x_test, self.y_test = self.model_obj.prepare_data(df, self.length)

        # Start the timer
        self.start_time = datetime.datetime.now()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)  # update every second

        self.thread = WorkerThread(lambda: self.run_test())
        self.thread.task_done.connect(self.on_task_done)
        self.thread.start()
        time.sleep(0.1)

    def run_test(self):
        """
        Run the test function in a new thread.
        :return: None
        """
        try:
            test(
                self.x_test, self.y_test, self.model_type.currentText(), self.model_obj
            )
        except FileNotFoundError:
            print("cannot load model for testing")

        self.is_done = True
        self.update_timer()  # update one last time when testing is done

    def open_file_dialog(self):
        """
        Open a file dialog to select the training data.
        :return: None
        """
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Training Data", "", "CSV files (*.csv)"
        )
        if file_name:
            self.selected_file_label.setText(os.path.basename(file_name))
            with open("settings.json", "w") as f:
                json.dump({"training_data": file_name}, f)

    def save_training_time(self, model_type, training_time):
        """
        Save the training time to a file.
        :param model_type: type of model
        :param training_time: time taken to train the model
        :return:
        """
        with open(
            get_results_location(model_type, self.model_obj.model_name), "a"
        ) as f:
            f.write(f"Training Time: {training_time} seconds\n")

    def train(self):
        """
        Train the model on prepared dataset.
        """
        print("Training model...")

        start_time = time.time()
        self.model_obj.train_model(self.x_train, self.y_train)
        training_time = time.time() - start_time
        print(f"Training time: {training_time} seconds.")

        # write training time to file
        self.save_training_time(self.model_type.currentText(), training_time)

        # Save the trained model
        joblib.dump(
            self.model_obj.model,
            get_model_location(
                self.model_type.currentText(), self.model_obj.model_name
            ),
        )

    def go_to_training_page(self):
        """your existing code"""
        self.stacked_widget.setCurrentIndex(1)
        self.make_model()
        df = load_dataset(self.selected_file_label.text())

        self.x_train, self.y_train = self.model_obj.prepare_data(df)
        # ...
        self.thread = WorkerThread(lambda: self.train())
        self.thread.task_done.connect(self.on_task_done)
        self.thread.start()
        time.sleep(0.1)

        # start timer
        self.start_time = datetime.datetime.now()

        # start timer to update label
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)  # update every second
        # self.render(self.model, trained_model, x, y)

    # *********************************************************
    def validate(self, k=10, data_length=None):
        """
        Perform cross-validation on dataset.
        Dataset is divided into k subsets,
        with k-1 used to train, and 1 left to validate on.

        :param data_length: Length of data to use.
        If None, use all data.
        :param k: Number of subdivisions to make from the dataset.
        :return: None.
        Saves model and results to model folder.
        """
        df = load_dataset(resample_test_filename)
        x_test, y_test = self.model_obj.prepare_data(df, data_length)

        size = int(len(self.x_test) / k)
        print("k-size:" + str(size))

        for i in range(k):
            print("subset " + str(i + 1))
            idx = i * size
            self.x_train, self.y_train, self.x_test, self.y_test = k_fold_xy(
                x_test, y_test, idx, size
            )

            value = self.x_train.isnull().sum().sum()
            if value > 0:
                print("null x vals: " + str(value))
                for col in self.model_obj.attributes:
                    num = x_train.isnull().sum()
                    print(str(num) + " null in " + col)
                return

            # Create a new worker thread for each iteration
            thread = WorkerThread(lambda: self.train_and_test())
            self.threads.append(thread)
            thread.task_done.connect(self.on_task_done)
            thread.finished.connect(lambda: self.remove_thread(thread))  # new line
            thread.start()

            # Wait for the thread to finish before starting the next one

        # Q: how do you check the size of a list?

        while len(self.threads) > 0:
            for thread in self.threads:
                if thread.isRunning():
                    thread.wait()
                else:
                    self.remove_thread(thread)
            self.update_timer()
            QApplication.processEvents()
            time.sleep(0.1)

        self.is_done = True

    def remove_thread(self, thread):  # new method
        try:
            self.threads.remove(thread)
        except ValueError:
            pass

    def train_and_test(self):
        """
        Train the model and perform a test.
        """
        self.make_model()  # New line to create a new model for each thread

        print("Training model...")
        start_time = time.time()
        self.model_obj.train_model(self.x_train, self.y_train)
        training_time = time.time() - start_time
        print(f"Training time: {training_time} seconds.")
        self.save_training_time(self.model_type.currentText(), training_time)
        test(self.x_test, self.y_test, self.model_type.currentText(), self.model_obj)

    def update_timer(self):
        """
        Update the timer label.
        :return: None
        """
        elapsed_time = datetime.datetime.now() - self.start_time
        if (self.thread is not None and self.thread.is_running) or len(
            self.threads
        ) > 0:
            self.complete_label.setText(
                f"Running: {str(elapsed_time.total_seconds())} seconds"
            )
            self.status_label.setText(
                f"Running: {str(elapsed_time.total_seconds())} seconds"
            )
        else:
            self.timer.stop()
            # Stop the timer

            self.status_label.setText(
                f"Complete: {str(elapsed_time.total_seconds())} seconds"
            )
            self.complete_label.setText(
                f"Complete: {str(elapsed_time.total_seconds())} seconds"
            )

    def on_task_done(self):
        """
        When the training task is done, update the timer one last time.
        :return: None
        """
        self.is_done = True
        self.update_timer()  # update one last time when training is done
        if self.thread is not None:
            self.thread.is_running = False


# noinspection PyUnresolvedReferences
class WorkerThread(QThread):
    """
    A thread to run a task in the background.
    """

    task_done = pyqtSignal()

    def __init__(self, task=None):
        """
        Initialize the thread.
        :param task: The task to run.
        """
        super().__init__()
        self.task = task
        self.is_running = False
        self.is_task_done = False

    def run(self):
        """
        Run the task.
        :return: None
        """
        self.is_running = True
        try:
            if self.task:
                self.task()
        finally:
            self.is_running = False
            self.is_task_done = True
            self.task_done.emit()
