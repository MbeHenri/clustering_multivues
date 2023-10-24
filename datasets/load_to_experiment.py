from scipy.io import loadmat
from numpy import array, reshape
from .load import load_msrcv1 as load_msrcv1_, load_multiple_features as load_multiple_features_


# chargement de la dataset msrcv1 pour les expérimentations d'un modèle
def load_msrcv1(to="mcles", path_dir_datasets=""):

    dataset = load_msrcv1_(path_dir_datasets=path_dir_datasets)
    
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
        
# chargement de la dataset multiples futures pour les expérimentations d'un modèle
def load_multiple_features(to="mcles", path_dir_datasets=""):
    dataset = load_multiple_features_(path_dir_datasets=path_dir_datasets)
    
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

# chargement de toutes les datasets pour les expérimentations d'un modèle
def load_multiple_features(to="mcles", path_dir_datasets=""):
    datasets = []
    datasets.append(load_msrcv1(to=to, path_dir_datasets=path_dir_datasets))
    datasets.append(load_multiple_features(to=to, path_dir_datasets=path_dir_datasets))
    
    return datasets