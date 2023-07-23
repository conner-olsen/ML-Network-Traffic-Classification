# test_SVM.py

"""
    author: em
    project: group 7 network ids training model
    class: CS-534 WPI
    last update: July 5, 2023

    file to test class implementation and integration for svm model
"""
from Models.SVM import SVM
from Util.Eval import evaluate_model
from Util.Util import load_dataset

DATA_ROOT = "Data/datasets/CIDDS/"
train_filename = DATA_ROOT + "training/CIDDS_Internal_train.csv"
attributes = ['Duration', 'Src_IP', 'Src_Pt', 'Dst_Pt', 'Packets', 'Flags']


def main():
    attr = ['Dst_Pt', 'Src_IP', 'Bytes', 'Label']    # subset of attr to train on
    svm_model = SVM(attr, 'svm_test')
    df = load_dataset(train_filename)
    x, y = svm_model.prepare_data(df, attributes, 30000)    # total to use
    print("model trained!")
    x_test = x.iloc[20000:30000]
    y_test = y.iloc[20000:30000]
    predict_svm = svm_model.test_model(x_test)
    print("model tested!")
    evaluate_model(x_test, y_test, "SVM", svm_model.get_model_name(), predict_svm)
    svm_model.render_model()


if __name__ == "__main__":
    main()
