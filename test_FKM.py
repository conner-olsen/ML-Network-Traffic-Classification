from FKM.FKM import *
from Utilities.prepare_data import load_dataset

filename = "Data/datasets/CIDDS/training/CIDDS_Internal_train.csv"


def test_FKM():
    fkm = FKM()

    df = load_dataset(filename)
    X, y = fkm.prepare_data(df)
    #print(X)

    '''
    for i in range(len(X[1])):
        val = type(X[1][i])
        if (val == int):
            print(str(val) +"==int")
        if (val == float):
            print(str(val) + "==float")
        if (val == str):
            print(str(val)+"==str")
        else:
            print(str(val))'''

    #fkm.random_grouping(X)

    #fkm.measure_dist()
    
    k = 2
    while k < 8:
        name = "fkmC"+str(k)
        tmp_fkm = FKM(k, name)
        tmp_fkm.train_model(X, y)
        tmp_fkm.render_model()
        k+=1

#if __name__=="__main__":
#    main()