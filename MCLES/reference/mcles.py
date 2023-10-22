from oct2py import octave
from sklearn.cluster import KMeans


def mcles(X, k, d=45, maxIters=30, alpha=0.8, beta=0.5, gamma=0.004, epsilon=0.01, maxItersForKmeans=1000, nInitForKmeans=20, verbose=False, path_ref_dir = './MCLES-master/MCLES Code/'):
    
    octave.addpath(octave.genpath(path_ref_dir))
    W, H, S, P, obj = octave.feval(
        "MCLES.m", X, k, alpha, beta, d, gamma, maxIters)
    octave.restart()

    # affichage de la fonction de perte
    if verbose:
        for it in range(len(obj)):
            print("iterations {} : Objective function : {}".format(
                it, obj[it]))
    # attribution des observations aux clusters
    kmeans = KMeans(
        n_clusters=k,
        max_iter=maxItersForKmeans,
        n_init=nInitForKmeans,
    )
    kmeans.fit(P)
    labels_clusters = kmeans.labels_
    return {"W": W, "H": H, "S": S, "P": P.real, "labels": labels_clusters, "objs": obj}

        