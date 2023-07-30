# main.py

"""
    author: co
    project: group 7 network ids model training
    class: CS-534 WPI
    last update: July 25, 2023

    main file to run the program
"""
import sys

from PyQt6.QtWidgets import QApplication

from Interface.MainWindow import MainWindow


# **********************************************************
def main():
    """
    Main function to control program flow.
    """

    # BEGIN USER INTERFACE
    app = QApplication([])
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
