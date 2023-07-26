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


    # BEGIN SELECTION OF MODELS - model type may impact data processing
    opt = train_or_test()
    if opts[opt] == "QUIT":
        print("quitting application...")
        return
    print("selected option: " + opts[opt])

    if opts[opt] != "PROCESS DATA":
        # prompt user to select the type of model to work with
        model_type = get_model_type()

        # get the model name to work with (may exist, if not model obj will be named)
        model_name = get_model_name(model_type)

        if model_type == "DT":
            model = DT(model_name)
        elif model_type == "SVM":
            dflt_svm_attr = ["Dst_Pt", "Src_IP", "Bytes", "Label"]
            model = SVM(dflt_svm_attr, model_name=model_name)
            # length = 50000 #demo length - SVM training is long
        elif model_type == "FKM":
            convert_str = False
            resample = False
            print("not implemented yet")
        elif model_type == "TACGAN":
            print("not implemented yet")
            return
        else:
            print("invalid option, exiting application")
            return
    # TODO: add FKM and TACGAN

    # BEGIN CLEANING/NORMALIZATION/TRAIN AND TEST SPLIT OF RAW DATA

    # INSTANTIATE THE DATAPREP CLASS
    data_opt = DataPrep(raw_data_path, DATA_ROOT)

    # IF USER HAS A RAW CSV TO PARSE
    data_opt.set_raw_dir()

    if data_opt.get_raw_dir():

        # prompt user to see if they want to convert strings to numbers
        convert_str = (
            input(
                "would you like to convert strings in the dataset into numeric values? (y/n): "
            )
            == "y"
        )
        resample = input("would you like to balance the dataset? (y/n): ") == "y"

        # BEGIN PARSING THE DATA
        data_opt.set_parse_data(convert_strings=convert_str)

        # BEGIN SPLITTING THE DATA
        data_opt.split_data(resample=resample)

    # Train, k-folds, or test
    if opts[opt] == "TRAIN":  # train
        if resample:
            df = load_dataset(resample_train_filename)
        else:
            df = load_dataset(train_filename)
        x, y = model.prepare_data(df, attributes, length)
        trained_model = train(x, y, model, model_type)
        render(model, trained_model, x, y)
    elif opts[opt] == "K-FOLD TRAIN & VALIDATE":  # cross validation routine
        k_fold_train_and_validate(10, model_type, train_filename, model, length)
    elif opts[opt] == "TEST":  # test
        if resample:
            df = load_dataset(resample_test_filename)
        else:
            df = load_dataset(test_filename)
        x, y = model.prepare_data(df, attributes, length)
        try:
            trained_model = load_saved_model(model_type, model_name)
            test(x, y, model_type, model)
        except:
            print("cannot load model for testing")
    elif opts[opt] == "PROCESS DATA":
        print("Data files ready")
    else:
        print("invalid selection, exiting application")


if __name__ == "__main__":
    main()
