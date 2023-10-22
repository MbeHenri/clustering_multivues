from oct2py import octave
from scipy.sparse.csgraph import connected_components

def mvgl(X, k, beta=1, gamma=1, hand_first_k_neighbour=True, hand_graph_neighbour=True, epsilon=1e-3, verbose=False, path_ref_dir="./MVGL-master"):
    
    octave.addpath(octave.genpath(path_ref_dir))
    Sv, A, P, obj = octave.feval("MVGL.m", X, k, beta, gamma,1 if hand_first_k_neighbour else 0, 1 if hand_graph_neighbour else 0, epsilon)
    octave.restart()
    
    Sv = Sv[0]
    A = A[0]
    P = P[0]
    obj = obj[0][0]

    # affichage de la fonction de perte
    if verbose:
        for it in range(len(obj)):
            print("iterations {} : Objective function : {}".format(
                it, obj[it]))
                
    # attribution des observations aux clusters
    _,labels=  connected_components((A+A.T)/2)
    
    return {"A": A, "P": P, "objs": obj, "labels": labels}

        