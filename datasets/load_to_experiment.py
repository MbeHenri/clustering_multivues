from scipy.io import loadmat
from numpy import array, reshape
from .load import load_msrcv1 as load_msrcv1_, load_multiple_features as load_multiple_features_



def load_msrcv1(to="mcles", path_dir_dataset=""):

    dataset = load_msrcv1_(path_dir_dataset=path_dir_dataset)
    
    if to == "mcles":
        
        # paramètres par défaut pour le modèle mcles
        dataset["parameters"] = {
            "maxIters": 10,
            "alpha": 0.8,
            "beta": 0.4,
            "d": 70,
            "gamma": 0.004,
            "k": 7,
        }
    
    elif to == "mvgl":
    
        # paramètres par défaut pour le modèle mvgl
        dataset["parameters"] = {
            "k": 7,
        }
    
        
    return dataset
        

def load_multiple_features(to="mcles", path_dir_dataset=""):
    dataset = load_multiple_features_(path_dir_dataset=path_dir_dataset)
    
    if to == "mcles":
        
        # paramètres par défaut pour le modèle mcles
        dataset["parameters"] = {
            "maxIters": 10,
            "alpha": 0.8,
            "beta": 0.4,
            "d": 70,
            "gamma": 0.004,
            "k": 10,
        }
    
    elif to == "mvgl":
    
        # paramètres par défaut pour le modèle mvgl
        dataset["parameters"] = {
            "k": 10,
        }
    