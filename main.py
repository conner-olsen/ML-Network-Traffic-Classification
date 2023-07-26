# main.py

"""
    author: em, co
    project: group 7 network ids model training
    class: CS-534 WPI
    last update: July 5, 2023

    user interface file

    select what type of model to train or test, get metrics, save models, etc
"""
import sys

from PyQt6.QtWidgets import QApplication

from Interface.MainWindow import MainWindow

# **********************************************************
def main():
    """
    Main function to control program flow.
    """

    app = QApplication([])
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
