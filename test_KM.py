#test_FM.py

"""
    author: em
    project: group 7
    class: CS-534 Artificial Intelligence WPI
    date: July 23, 2023

    test implementation of KM model

"""
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from Models.KM import KM

filename = "Data/datasets/CIDDS/training/CIDDS_Internal_train.csv"
TRAIN_FULL = False


def main():
    k = 5           #num clusters

    df = pd.read_csv(filename)

    km_model = KM(k, "testKM")
    print(km_model.get_model_name())

    X, y = km_model.prepare_data(df)

    # full train set
    if TRAIN_FULL:
        x_train, x_test, y_train,  y_test = train_test_split(X, y, test_size=.33)
    else:
        # train /test a subset
        x_train = X.iloc[:20000]
        y_train = y.iloc[:20000]
        x_test = X.iloc[20000:30000]
        y_test = y.iloc[20000:30000]

    scaler = RobustScaler()
    scaler.fit_transform(x_train)
    scaler.transform(x_test)
    
    print("training...")
    
    start = time.time()
    trained = km_model.train_model(x_train, y_train)
    train_time = time.time() - start
    print("done training clusters! training time: " + str(train_time))

    print("testing...")
    start = time.time()
    predict = km_model.test_model(x_test)
    test_time = time.time() - start
    print("done testing! testing time: " + str(test_time))

    print("training scores")
    km_model.evaluate(x_train, y_train, trained.labels_)

    print()

    # print("testing scores")
    km_model.evaluate(x_test, y_test, predict)

    # print("training clustering visualized")
    km_model.render_model(predict, x_test, y_test)


if __name__ == "__main__":
    main()
