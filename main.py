#main.py

'''
    author: em, co
    project: group 7 network ids model training
    class: CS-534 WPI
    last update: July 5, 2023

    user interface file

    select what type of model to train or test, get metrics, save models, etc
'''
import time
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from Utilities.evaluate import *
from Utilities.prepare_data import *
from Utilities.utilities import *

from DT.DT import *
from SVM.SVM import *
from Data.DataPrep import *
from Utilities.KendallCorrelation import *
from Utilities.PearsonsCorrelation import *

DATA_ROOT = "Data/datasets/CIDDS/"
raw_data_path = "Data/datasets/CIDDS/CIDDS-001/"
train_filename = DATA_ROOT + "training/CIDDS_Internal_train.csv"
test_filename = DATA_ROOT + "testing/CIDDS_Internal_test.csv"
MODEL_ROOT = ['DT', 'FKM', 'SVM', 'TACGAN']
opts = ['TRAIN', 'K-FOLD TRAIN & VALIDATE', 'TEST']
attributes = ['Duration', 'Src_IP', 'Src_Pt', 'Dst_Pt', 'Packets', 'Flags']
parallel = joblib.Parallel(n_jobs=2, prefer="threads")

#********************************************************
def k_fold_xy(x, y, idx, size):
#
#   function to segment the k subsets of the data for
#       k-folds training and validation
#   Note: this is done in development on training set
#
#   params:
#       x: the prepared data to be partitioned
#       y: the prepared labels to be partitioned
#       idx: k-val * iteration
#       size: size of each k-subset
#   returns:
#       x_train, y_train, x_test, y_test: divided data/labels
#           for train and validate cycle (&evaluate)
#*********************************************************
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    front_x = x.iloc[:idx, :]         #start at 0-test subset idx
    back_x = x.iloc[idx+size:, :]     #start after set
    front_y = y.iloc[:idx] 
    back_y = y.iloc[idx+size:] 

    frames_x = [front_x, back_x]       #put together front and back
    frames_y = [front_y, back_y]
    x_train = pd.concat(frames_x)
    y_train = pd.concat(frames_y)

    x_test = x.iloc[idx:idx+size, :]    #prepare test section
    y_test = y.iloc[idx:idx+size]

    return x_train, y_train, x_test, y_test


#*********************************************************
def k_fold_train_and_validate(k, model_type, filename, model_obj, data_length=0):
#
#   function to perform cross validation on dataset
#       dataset is divided into k subsets, with k-1 used
#       to train, and 1 left to validate on
#
#   k: number of subdivisions to make from the dataset
#   model_type: type of model to work with (from user selection)
#   model_name: name of the model 
#   filename: data file to use for dataset
#
#   returns:
#       nothing, but saves model, saves results to model folder
#*********************************************************
    df = load_dataset(filename)
    x, y = model_obj.prepare_data(df, attributes, data_length)
    
    size = int(len(x)/k)
    print("k-size:"+str(size))

    for i in range(k):
        print("subset " + str(i+1))
        idx = i * size 
        x_train, y_train, x_test, y_test = k_fold_xy(x, y, idx, size)

        #check for null, then show where it is
        #this is just for debugging and to short circuit on bad data
        #maybe should move this to preparing data?
        value = x_train.isnull().sum().sum()
        if (value > 0):
            print("null x vals: " + str(value))
            for col in attributes:
                num = x_train.isnull().sum()
                print(str(num) + " null in " + col)
            return

        #train
        trained_model = train(x_train, y_train, model_obj, model_type)

        #testing
        #load the saved training model
        test(x_test, y_test, model_type, model_obj)
        
        render(model_obj, trained_model, x_train, y_train)

#***********************************************************
def render(model_obj, trained_model, x_train, y_train):
#
#   ask before rendering 
#
#***********************************************************
    #this is sticky, press enter after closing the render (in my experience)
    render = input("render model? (y/n): ")
    if (render == 'y'):
        model_obj.render_model(trained_model, x_train, y_train)


#**********************************************************
def train(x, y, model, model_type):
#
#   Train the model on prepared dataset
#
#   params:
#       x: x data (training)
#       y: y data (training - labels)
#       model: a model object
#       model_type: the type of model (where to save model)
#   calls:
#       model.train_model(x, y)
#       get_results_location: save training time
#       set_model_location: save the trained model
#   returns:
#       trained_model: the trained model
#***********************************************************
    print("Training model...")

    start_time = time.time()
    trained_model = model.train_model(x, y)
    training_time = time.time() - start_time
    print(f"Training time: {training_time} seconds.")
        
    #write training time to file
    with open(get_results_location(model_type, model.model_name), 'a') as f:
        f.write(f'Training Time: {training_time} seconds\n')
        
    # Save the trained model        
    joblib.dump(trained_model, set_model_location(model_type, model.model_name))

    return trained_model

#**************************************************************
def test(x_test, y_test, model_type, model_obj):
#
#   function to test the model after training
#
#   params:
#       x_test: test data (samples)
#       y_test: test data labels
#       model_type: type of model being trained
#       model_obj: model object
#   calls:
#       load_saved_model():loads trained model
#       model_obj.test_model(): tests the model (predict)
#       evaluate_model(): saves performance results to file
#   returns:
#       nothing, but saves data from evaluation
#**************************************************************
    trained_model = load_saved_model(model_type, model_obj.model_name)
    model_obj.set_model(trained_model)
    predictions = model_obj.test_model(x_test)
    evaluate_model(x_test, y_test, model_type, model_obj.model_name, predictions)


#***********************************************************
def train_or_test():
#
#   select to train, k-folds, or test with an existing model
#
#   returns:
#       choice (int): 0 for train, 1 for k-folds, 2 for test
#************************************************************ 
    while True:
        for i in range(len(opts)):
            print(str(i) + ': ' + opts[i])
        choice = input("Select an activity: ")
        if (int(choice) in [0, 1, 2]):    
            break
        else:
            print("Invalid entry, please try again")
    return int(choice)
    

#****************************************************************
def load_saved_model(model_type, model_name):
#
#   load a saved model, if it exists
#
#   params:
#       model_type: type of model
#       model_name: name of model
#   returns: 
#       model: saved model, or none if error
#***************************************************************
    model = None
    try:
        model = joblib.load(get_model_location(model_type, model_name))
        print("Loading model...")
    except:
        print("could not load model")

    return model


#****************************************************************
def get_model_name(model_type, opt):
#   
#   get the model name to use
#       if using an existing model, verify that the model exists
#       let user try again if the model was not found
#
#   params:
#       model_type: type of model to work with
#       opt: a number corresponding to the selection 
#           (0)- train (1)- k-folds (2)- test
#   return:
#       model_name: name of the model
#   
#   TODO: implement warm start to iteratively train the same model
#       (reason to allow loading an existing model with train opt)
#*****************************************************************
    use_existing = input("Use existing model? (y/n): ") == 'y'
    model_name = input("Enter model name (enter for default): ") 

    while True:
        if (use_existing):
            #test if it exists
            try:
                print(get_model_location(model_type, model_name))
                model_exist = get_model_location(model_type, model_name)
                break
            except:
                print("model does not exist")
        else:
            break

    return model_name


#*********************************************************************
def get_model_type():
#
#   user selects model type to work with
#
#   returns:
#       model_type (str): the type of model from selection menu
#*********************************************************************
    idx = -1
    for i in range(len(MODEL_ROOT)):
        print(str(i) + ': ' + MODEL_ROOT[i])
    
    while True:
        t = input("Select model type: ")
        idx = int(t)
        if (idx > len(MODEL_ROOT) or (idx < 0)):
            print("Invalid selection. Please try again")
        else:
            break
    return MODEL_ROOT[idx]


#**********************************************************
def main():
#
#   main function to control program flow
#   
#***********************************************************

    ## BEGIN CLEANING/NORMALIZATION/TRAIN AND TEST SPLIT OF RAW DATA

    ## INSTANTIATE THE DATAPREP CLASS
    data_opt = DataPrep(raw_data_path, DATA_ROOT)

    ## IF USER HAS A RAW CSV TO PARSE
    data_opt.set_raw_dir()

    if data_opt.get_raw_dir():

        ## BEGIN PARSING THE DATA
        data_opt.set_parse_data()

        ## BEGIN SPLITTING THE DATA
        data_opt.split_data()

    ## IF DATA HAS ALREADY BEEN PARSED
    else:
        pass

    ## BEGIN SELECTION OF MODELS
    model = None
    length = -1     #default to use full dataset in training/testing

    opt = train_or_test()

    #prompt user to select the type of model to work with
    model_type = get_model_type()

    #get the model name to work with (may exist, if not model obj will be named)
    model_name = get_model_name(model_type, opt)

    if (model_type == "DT"):
        model = DT(model_name)
    elif (model_type == "SVM"):
        model = SVM(['Dst_Pt', 'Src_IP', 'Bytes', 'Label'], model_name)
        length = 50000 #demo length - SVM training is long
    else:
        print("not implemented yet")
        return
    #TODO: add FKM and TACGAN 
    
    if (opt == 0):  #train
        df = load_dataset(train_filename)  
        x, y = model.prepare_data(df, attributes, length) 
        trained_model = train(x, y, model, model_type)
        render(model, trained_model, x, y)
    elif (opt == 1):    #cross validation routine
        k_fold_train_and_validate(10, model_type, train_filename, model, length)    
    else:   #test
        df = load_dataset(test_filename)
        x, y = model.prepare_data(df, attributes, length)
        try:
            trained_model = load_saved_model(model_type, model_name)
            test(x, y, model_type, model)
        except:
            print("cannot load model for testing")


if __name__ == "__main__":
    main()
