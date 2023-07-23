import json
import os

from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFileDialog, QPushButton, \
    QLabel, QComboBox, QStackedWidget, QLineEdit

from Models.DT import DT
from Models.SVM import SVM
from Util.Util import load_dataset, prepare_data
from main import train_filename, attributes, train, render


# noinspection PyUnresolvedReferences
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = None
        self.model_type = None
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
