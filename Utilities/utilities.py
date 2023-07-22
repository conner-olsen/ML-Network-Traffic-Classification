import os

#*********************************************************
@staticmethod
def get_results_location(models_folder, model_name: str):
#   
#   Get the location of the results file for saving test results
#       if folder and file do not exist, they are created
#    
#   Params:
#       model_name (str): the name of the file to save the model to
#   Returns:
#       str: the location of the results file
#*************************************************************
    location = os.path.join(models_folder + '/results/', model_name)
    os.makedirs(location, exist_ok=True)
    return os.path.join(location, 'results.txt')


#**************************************************************
@staticmethod
def get_model_location(models_folder, model_name: str):
#
#   Retrieves path to a saved model in the format
#       <models_folder>/models/<model_name>/model.pkl
#       from current directory (should be called from main)
#
#   Params:
#       model_name (str): a subfolder in <model type>/models/ 
#           to <model name>
#
#   Returns:
#       str: the relative path to the model from current dir
#       str = None if model does not exist
#**************************************************************
    location = os.path.join(models_folder+"/models/", model_name)
    if (os.path.isfile(os.path.join(location, 'model.pkl'))):
        os.makedirs(location, exist_ok=True)
        return os.path.join(location, 'model.pkl')
    else:
        print("file does not exist")
        return None

#*************************************************************
@staticmethod
def set_model_location(models_folder, model_name):
#
#   function to create the model in the models folder of the 
#       selected model type
#
#   params:
#       models_folder (str): the type of model we are training - 
#           this decides where the trained model is saved 
#       model_name: sub-folder under models is named for the current
#           tested model 
#   returns: 
#       model_path (str): relative path to the folder from project 
#           root
#****************************************************************
    location = os.path.join(models_folder+"/models/", model_name)
    os.makedirs(location, exist_ok=True)
    return os.path.join(location, 'model.pkl')
