# test_DT.py

"""
    author: em
    project: group 7 network ids model training
    class: CS-534 WPI
    last update: July 5, 2023

    file to test implementation and integration of the DT class
"""
from DT.DT import DT
from Utilities.eval import evaluate_model
from Utilities.prepare_data import load_dataset

DATA_ROOT = "Data/datasets/CIDDS/"
train_filename = DATA_ROOT + "training/CIDDS_Internal_train.csv"
attributes = ['Duration', 'Src_IP', 'Src_Pt', 'Dst_Pt', 'Packets', 'Flags']


def test_DT():
    dt_model = DT('dt_test')
    df = load_dataset(train_filename)
    x, y = prepare_data(df, attributes)
    x_train = x.iloc[:30000]    # demo train subset
    y_train = y.iloc[:30000]
    print("new x len" + str(len(x_train)))
    trained_dt = dt_model.train_model(x_train, y_train)
    print("model trained!")
    x_test = x.iloc[30000:50000]    # demo test (from train) subset
    y_test = y.iloc[30000:50000]
    print("test len:" + str(len(x_test)))
    predict_dt = dt_model.test_model(x_test)
    print("model tested!")
    evaluate_model(x_test, y_test, "DT", dt_model.get_model_name(), predict_dt)
    dt_model.render_model(trained_dt, x_train, y_train)
