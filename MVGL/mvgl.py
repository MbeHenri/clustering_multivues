from .reference.mvgl import mvgl as mvgl_ref
from .scratch.mvgl import mvgl as mvgl_sc


class MVGL():

    def __init__(self, beta=1, gamma=1, hand_first_k_neighbour=True, hand_graph_neighbour=True, epsilon=1e-3, path_ref_dir="./MVGL-master", scratch=True) -> None:

        self.beta = beta
        self.gamma = gamma
        self.hand_first_k_neighbour = hand_first_k_neighbour
        self.hand_graph_neighbour = hand_graph_neighbour
        self.epsilon = epsilon
        self.path_ref_dir = path_ref_dir
        self.scratch = scratch
        self.A, self.P, self.objs, self.labels = None, None, None, None, None

    def fit(self, X, k, verbose=False):

        if not self.scratch:
            result = mvgl_sc(
                X, k,
                beta=self.beta, 
                gamma=self.gamma,
                hand_first_k_neighbour=self.hand_first_k_neighbour,
                hand_graph_neighbour=self.hand_graph_neighbour,
                epsilon=self.epsilon,
                verbose=verbose,
            )
        else:
            pass
            
        
        self.A = result['A']
        self.P= result['P']
        self.objs= result['objs'] 
        self.labels= result['labels']
