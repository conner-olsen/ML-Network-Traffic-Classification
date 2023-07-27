import json
import os

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
)

from Models.DT import DT
from Models.SVM import SVM
from Util.Util import load_dataset, prepare_data

import sys
import time

import joblib
import pandas as pd
from PyQt6.QtWidgets import QApplication

from Data.DataPrep import DataPrep
from Models.DT import DT
from Models.SVM import SVM
from Util.Eval import evaluate_model
from Util.Util import (
    get_results_location,
    set_model_location,
    get_model_location,
    load_dataset,
    prepare_data,
)

DATA_ROOT = "Data/datasets/"
raw_data_path = "Data/datasets/CIDDS/CIDDS-001/"
train_filename = DATA_ROOT + "training/CIDDS_Internal_train.csv"
test_filename = DATA_ROOT + "testing/CIDDS_Internal_test.csv"
resample_test_filename = DATA_ROOT + "testing/CIDDS_Internal_test_resample_strings.csv"
resample_train_filename = (
    DATA_ROOT + "training/CIDDS_Internal_train_resample_strings.csv"
)
MODEL_ROOT = ["DT", "FKM", "SVM", "KM"]
opts = ["TRAIN", "K-FOLD TRAIN & VALIDATE", "TEST"]
attributes = ["Duration", "Src_IP", "Src_Pt", "Dst_Pt", "Packets", "Flags", "Label"]
default_svm_attr = ["Dst_Pt", "Src_IP", "Bytes", "Label"]
default_attr = ["Duration", "Src_IP", "Src_Pt", "Dst_Pt", "Packets", "Flags", "Label"]
parallel = joblib.Parallel(n_jobs=2, prefer="threads")


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


# *********************************************************
def k_fold_train_and_validate(k, model_type, filename, model_obj, data_length=0):
    """
    Perform cross-validation on dataset. Dataset is divided into k subsets,
    with k-1 used to train, and 1 left to validate on.

    :param data_length: Length of data to use (0 for all).
    :param model_obj: Model object.
    :param k: Number of subdivisions to make from the dataset.
    :param model_type: Type of model to work with (from user selection).
    :param filename: Data file to use for dataset.
    :return: None. Saves model and results to model folder.
    """
    df = load_dataset(filename)
    x, y = prepare_data(df, attributes, data_length)

    size = int(len(x) / k)
    print("k-size:" + str(size))

    for i in range(k):
        print("subset " + str(i + 1))
        idx = i * size
        x_train, y_train, x_test, y_test = k_fold_xy(x, y, idx, size)

        # check for null, then show where it is
        # this is just for debugging and to short circuit on bad data
        # maybe should move this to preparing data?
        value = x_train.isnull().sum().sum()
        if value > 0:
            print("null x vals: " + str(value))
            for col in attributes:
                num = x_train.isnull().sum()
                print(str(num) + " null in " + col)
            return

        # train
        trained_model = train(x_train, y_train, model_obj, model_type)

        # testing
        # load the saved training model
        test(x_test, y_test, model_type, model_obj)

        render(model_obj, trained_model, x_train, y_train)


# ***********************************************************
def render(model_obj, trained_model, x_train, y_train):
    """
    Ask before rendering.

    :param model_obj: Model object.
    :param trained_model: Trained model.
    :param x_train: Training data.
    :param y_train: Training labels.
    """
    render_model = "y"
    if render_model == "y":
        model_obj.render_model(trained_model, x_train, y_train)


# **********************************************************
def train(x, y, model, model_type):
    """
    Train the model on prepared dataset.

    :param x: Training data.
    :param y: Training labels.
    :param model: Model object.
    :param model_type: Type of model (where to save model).
    :return: Trained model.
    """
    print("Training model...")

    start_time = time.time()
    trained_model = model.train_model(x, y)
    training_time = time.time() - start_time
    print(f"Training time: {training_time} seconds.")

    # write training time to file
    with open(get_results_location(model_type, model.model_name), "a") as f:
        f.write(f"Training Time: {training_time} seconds\n")

    # Save the trained model
    joblib.dump(trained_model, set_model_location(model_type, model.model_name))

    return trained_model


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
    trained_model = load_saved_model(model_type, model_obj.model_name)
    model_obj.set_model(trained_model)
    predictions = model_obj.test_model(x_test)
    evaluate_model(x_test, y_test, model_type, model_obj.model_name, predictions)


# ****************************************************************
def load_saved_model(model_type, model_name):
    """
    Load a saved model, if it exists.

    :param model_type: Type of model.
    :param model_name: Name of model.
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
    def __init__(self):
        super().__init__()
        self.settings = None
        self.model_type = None
        self.resample = True
        self.convert_str = False
        self.trained_model = None
        self.model = None
        self.samples = None
        self.labels = None
        self.length = -1
        self.selected_file_label = None
        self.file_dialog = None
        self.output_textbox = None
        self.setWindowTitle("CS534 Final Project")

        self.stacked_widget = QStackedWidget()  # Stacked widget

        self.create_main_menu()
        self.create_training_page()
        self.create_validating_page()
        self.create_testing_page()

        # Set the stacked widget as the central widget
        self.setCentralWidget(self.stacked_widget)

    def create_main_menu(self):
        layout = QVBoxLayout()  # Main menu layout

        # ComboBox
        self.model_type = QComboBox()
        self.model_type.addItems(MODEL_ROOT)
        layout.addWidget(QLabel("Select the model type"))
        layout.addWidget(self.model_type)

        # File Dialog
        self.file_dialog = QPushButton("Select Training Data")
        self.file_dialog.clicked.connect(self.open_file_dialog)
        self.selected_file_label = QLabel("No file selected")
        layout.addWidget(self.file_dialog)
        layout.addWidget(self.selected_file_label)

        # Loading settings
        try:
            with open("settings.json", "r") as f:
                self.settings = json.load(f)
                self.selected_file_label.setText(
                    os.path.basename(self.settings["training_data"])
                )
        except FileNotFoundError:
            print("File 'settings.json' not found")
        except json.JSONDecodeError:
            print("Error decoding the JSON file")
        except KeyError:
            print("Key 'training_data' not found in the JSON file")

        # Create a new QLineEdit widget
        self.output_textbox = QLineEdit()

        # Add the textbox to your layout
        layout.addWidget(self.output_textbox)

        # Training, Validating, Testing buttons
        hbox = QHBoxLayout()
        prepare_button = QPushButton("Prepare")
        train_button = QPushButton("Training")
        validate_button = QPushButton("Validating")
        test_button = QPushButton("Testing")

        # Connect button click signals to the appropriate
        prepare_button.clicked.connect(self.prepare_data)
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
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Training Complete"))
        render_button = QPushButton("Render")
        render_button.clicked.connect(self.render_trained)
        layout.addWidget(render_button)
        back_button = QPushButton("Back to Main Menu")
        back_button.clicked.connect(self.go_to_main_menu)
        layout.addWidget(back_button)

        widget = QWidget()
        widget.setLayout(layout)

        self.stacked_widget.addWidget(widget)

    def create_validating_page(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Complete"))
        back_button = QPushButton("Back to Main Menu")
        back_button.clicked.connect(self.go_to_main_menu)
        layout.addWidget(back_button)

        widget = QWidget()
        widget.setLayout(layout)

        self.stacked_widget.addWidget(widget)

    def render_trained(self):
        self.model.render_model(self.trained_model, self.samples, self.labels)

    def create_testing_page(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Complete"))
        back_button = QPushButton("Back to Main Menu")
        back_button.clicked.connect(self.go_to_main_menu)
        layout.addWidget(back_button)

        widget = QWidget()
        widget.setLayout(layout)

        self.stacked_widget.addWidget(widget)

    def go_to_main_menu(self):
        self.stacked_widget.setCurrentIndex(0)

    def make_model(self):
        """
        Make the model object based on the selected model type
        :return: None
        """
        global attributes
        self.length = -1
        model_type = self.model_type.currentText()

        attributes = default_attr
        if model_type == "DT":
            self.model = DT(self.output_textbox.text())
        elif model_type == "SVM":
            attributes = default_svm_attr
            self.model = SVM(attributes, self.output_textbox.text())
            # length = 50000 #demo length - SVM training is long
        elif model_type == "FKM":
            self.convert_str = False
            self.resample = False
            print("not implemented yet")
        elif model_type == "TACGAN":
            print("not implemented yet")
            return
        else:
            print("invalid option, exiting application")
            return

    def prepare_data(self):
        self.make_model()

        # BEGIN CLEANING/NORMALIZATION/TRAIN AND TEST SPLIT OF RAW DATA

        # INSTANTIATE THE DATAPREP CLASS
        data_opt = DataPrep(raw_data_path, DATA_ROOT)

        # IF USER HAS A RAW CSV TO PARSE
        data_opt.set_raw_dir()

        if data_opt.get_raw_dir():
            data_opt.set_parse_data(self.convert_str)
            data_opt.split_data(self.resample)

    def go_to_training_page(self):
        self.stacked_widget.setCurrentIndex(1)

        self.make_model()

        df = load_dataset(self.settings["training_data"])

        self.samples, self.labels = prepare_data(df, attributes, self.length)
        self.trained_model = train(
            self.samples, self.labels, self.model, self.model_type.currentText()
        )
        # self.render(self.model, trained_model, x, y)

    def go_to_validating_page(self):
        self.stacked_widget.setCurrentIndex(2)

        self.make_model()

        k_fold_train_and_validate(
            10,
            self.model_type.currentText(),
            self.settings["training_data"],
            self.model,
            self.length,
        )

    def go_to_testing_page(self):
        self.stacked_widget.setCurrentIndex(3)

        self.make_model()

        if self.resample:
            df = load_dataset(resample_test_filename)
        else:
            df = load_dataset(test_filename)
        x, y = prepare_data(df, attributes, self.length)
        try:
            # trained_model = load_saved_model(self.model_type.currentText(), model_name)
            test(x, y, self.model_type.currentText(), self.model)
        except FileNotFoundError:
            print("cannot load model for testing")

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Training Data", "", "CSV files (*.csv)"
        )
        if file_name:
            self.selected_file_label.setText(os.path.basename(file_name))
            with open("settings.json", "w") as f:
                json.dump({"training_data": file_name}, f)

    def closeEvent(self, event):
        with open("settings.json", "w") as f:
            json.dump(self.settings, f)
