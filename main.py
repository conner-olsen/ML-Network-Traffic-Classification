# main.py

"""
    author: em, co
    project: group 7 network ids model training
    class: CS-534 WPI
    last update: July 5, 2023

    user interface file

    select what type of model to train or test, get metrics, save models, etc
"""
import os
import time
import joblib
import pandas as pd
import sys
import json
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFileDialog, QPushButton,\
    QLabel, QComboBox, QStackedWidget, QLineEdit
from PyQt6.QtCore import Qt

from Data.DataPrep import DataPrep

from Models.DT import DT
from Models.SVM import SVM

from Util.Eval import evaluate_model
from Util.Util import get_results_location, set_model_location, get_model_location, load_dataset, prepare_data

DATA_ROOT = "Data/datasets/CIDDS/"
raw_data_path = "Data/datasets/CIDDS/CIDDS-001/"
train_filename = DATA_ROOT + "training/CIDDS_Internal_train.csv"
test_filename = DATA_ROOT + "testing/CIDDS_Internal_test.csv"
MODEL_ROOT = ['DT', 'FKM', 'SVM', 'TACGAN']
opts = ['TRAIN', 'K-FOLD TRAIN & VALIDATE', 'TEST']
attributes = ['Duration', 'Src_IP', 'Src_Pt', 'Dst_Pt', 'Packets', 'Flags']
parallel = joblib.Parallel(n_jobs=2, prefer="threads")


def k_fold_xy(x, y, idx, size):
    """
    Segment the k subsets of the data for k-folds training and validation.
    Note: this is done in development on training set.

    :param x: The prepared data to be partitioned.
    :param y: The prepared labels to be partitioned.
    :param idx: k-val * iteration.
    :param size: Size of each k-subset.
    :return: Divided data/labels for train and validate cycle (&evaluate).
    """

    front_x = x.iloc[:idx, :]  # start at 0-test subset idx
    back_x = x.iloc[idx + size:, :]  # start after set
    front_y = y.iloc[:idx]
    back_y = y.iloc[idx + size:]

    frames_x = [front_x, back_x]  # put together front and back
    frames_y = [front_y, back_y]
    x_train = pd.concat(frames_x)
    y_train = pd.concat(frames_y)

    x_test = x.iloc[idx:idx + size, :]  # prepare test section
    y_test = y.iloc[idx:idx + size]

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
    render_model = input("render model? (y/n): ")
    if render_model == 'y':
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
    with open(get_results_location(model_type, model.model_name), 'a') as f:
        f.write(f'Training Time: {training_time} seconds\n')

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


# ***********************************************************
def train_or_test():
    """
    Select to train, k-folds, or test with an existing model.

    :return: Choice (int): 0 for train, 1 for k-folds, 2 for test.
    """
    while True:
        for i in range(len(opts)):
            print(str(i) + ': ' + opts[i])
        choice = input("Select an activity: ")
        if int(choice) in [0, 1, 2]:
            break
        else:
            print("Invalid entry, please try again")
    return int(choice)


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


# ****************************************************************
def get_model_name(model_type):
    """
    get the model name to use
    if using an existing model, verify that the model exists
    let user try again if the model was not found
    :param model_type: type of model to work with
    :return: name of the model
    """
    use_existing = input("Use existing model? (y/n): ") == 'y'
    model_name = input("Enter model name (enter for default): ")

    while True:
        if use_existing:
            # test if it exists
            try:
                print(get_model_location(model_type, model_name))
                break
            except FileNotFoundError:
                print("model does not exist")
        else:
            break

    return model_name


# *********************************************************************
def get_model_type():
    #
    #   user selects model type to work with
    #
    #   returns:
    #       model_type (str): the type of model from selection menu
    # *********************************************************************
    for i in range(len(MODEL_ROOT)):
        print(str(i) + ': ' + MODEL_ROOT[i])

    while True:
        t = input("Select model type: ")
        idx = int(t)
        if idx > len(MODEL_ROOT) or (idx < 0):
            print("Invalid selection. Please try again")
        else:
            break
    return MODEL_ROOT[idx]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
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
        self.model_type.addItems(["DT", "FKM", "SVM", "TACGAN"])
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
                self.selected_file_label.setText(os.path.basename(self.settings["training_data"]))
        except Exception:
            pass

        # Create a new QLineEdit widget
        self.output_textbox = QLineEdit()

        # Add the textbox to your layout
        layout.addWidget(self.output_textbox)

        # Training, Validating, Testing buttons
        hbox = QHBoxLayout()
        train_button = QPushButton("Training")
        validate_button = QPushButton("Validating")
        test_button = QPushButton("Testing")

        # Connect button click signals to the appropriate slots
        train_button.clicked.connect(self.go_to_training_page)
        validate_button.clicked.connect(self.go_to_validating_page)
        test_button.clicked.connect(self.go_to_testing_page)

        hbox.addWidget(train_button)
        hbox.addWidget(validate_button)
        hbox.addWidget(test_button)

        layout.addLayout(hbox)

        widget = QWidget()  # Wrap layout in a QWidget
        widget.setLayout(layout)

        self.stacked_widget.addWidget(widget)  # Add widget to the stack

    def create_training_page(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Training Page"))
        back_button = QPushButton("Back to Main Menu")
        back_button.clicked.connect(self.go_to_main_menu)
        layout.addWidget(back_button)

        widget = QWidget()
        widget.setLayout(layout)

        self.stacked_widget.addWidget(widget)

    def create_validating_page(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Validating Page"))
        back_button = QPushButton("Back to Main Menu")
        back_button.clicked.connect(self.go_to_main_menu)
        layout.addWidget(back_button)

        widget = QWidget()
        widget.setLayout(layout)

        self.stacked_widget.addWidget(widget)

    def create_testing_page(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Testing Page"))
        back_button = QPushButton("Back to Main Menu")
        back_button.clicked.connect(self.go_to_main_menu)
        layout.addWidget(back_button)

        widget = QWidget()
        widget.setLayout(layout)

        self.stacked_widget.addWidget(widget)

    def go_to_main_menu(self):
        self.stacked_widget.setCurrentIndex(0)

    def go_to_training_page(self):
        self.stacked_widget.setCurrentIndex(1)

        model_type = self.model_type.currentText()
        length = -1  # default to use full dataset in training/testing

        if model_type == "DT":
            model = DT(self.output_textbox.text())
        elif model_type == "SVM":
            model = SVM(['Dst_Pt', 'Src_IP', 'Bytes', 'Label'], self.output_textbox.text())
            length = 50000  # demo length - SVM training is long
        else:
            print("not implemented yet")
            return

        df = load_dataset(train_filename)
        x, y = prepare_data(df, attributes, length)
        trained_model = train(x, y, model, model_type)
        render(model, trained_model, x, y)

    def go_to_validating_page(self):
        self.stacked_widget.setCurrentIndex(2)

    def go_to_testing_page(self):
        self.stacked_widget.setCurrentIndex(3)

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Training Data", "", "CSV files (*.csv)")
        if file_name:
            self.selected_file_label.setText(os.path.basename(file_name))
            with open("settings.json", "w") as f:
                json.dump({"training_data": file_name}, f)

    def closeEvent(self, event):
        with open('settings.json', 'w') as f:
            json.dump(self.settings, f)

# **********************************************************
def main():
    """
    Main function to control program flow.
    """

    app = QApplication([])
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

    # BEGIN CLEANING/NORMALIZATION/TRAIN AND TEST SPLIT OF RAW DATA

    # INSTANTIATE THE DATAPREP CLASS
    data_opt = DataPrep(raw_data_path, DATA_ROOT)

    # IF USER HAS A RAW CSV TO PARSE
    data_opt.set_raw_dir()

    if data_opt.get_raw_dir():

        # BEGIN PARSING THE DATA
        data_opt.set_parse_data()

        # BEGIN SPLITTING THE DATA
        data_opt.split_data()

    # IF DATA HAS ALREADY BEEN PARSED

    # BEGIN SELECTION OF MODELS
    length = -1  # default to use full dataset in training/testing

    opt = train_or_test()

    # prompt user to select the type of model to work with
    model_type = get_model_type()

    # get the model name to work with (may exist, if not model obj will be named)
    model_name = get_model_name(model_type)

    if model_type == "DT":
        model = DT(model_name)
    elif model_type == "SVM":
        model = SVM(['Dst_Pt', 'Src_IP', 'Bytes', 'Label'], model_name)
        length = 50000  # demo length - SVM training is long
    else:
        print("not implemented yet")
        return
    # TODO: add FKM and TACGAN

    if opt == 0:  # train
        df = load_dataset(train_filename)
        x, y = prepare_data(df, attributes, length)
        trained_model = train(x, y, model, model_type)
        render(model, trained_model, x, y)
    elif opt == 1:  # cross validation routine
        k_fold_train_and_validate(10, model_type, train_filename, model, length)
    else:  # test
        df = load_dataset(test_filename)
        x, y = prepare_data(df, attributes, length)
        try:
            load_saved_model(model_type, model_name)
            test(x, y, model_type, model)
        except FileNotFoundError:
            print("cannot load model for testing")


if __name__ == "__main__":
    main()
