from .reference.mcles import mcles as mcles_ref
from .scratch.mcles import mcles as mcles_sc


class MCLES():

    def __init__(self, d=45, maxIters=30, alpha=0.8, beta=0.5, gamma=0.004, epsilon=0.01, maxItersForKmeans=1000, nInitForKmeans=20, path_ref_dir='./MCLES-master/MCLES Code/', scratch=True) -> None:
        self.d = 45
        self.maxIters = 30
        self.alpha = 0.8
        self.beta = 0.5
        self.gamma = 0.004
        self.epsilon = 0.01
        self.maxItersForKmeans = 1000
        self.nInitForKmeans = 20
        self.W, self.H, self.S, self.P, self.objs = None, None, None, None, None
        self.path_ref_dir = path_ref_dir
        self.scratch = scratch

    def fit(self, X, k, verbose=False):

        if not self.scratch:
            result = mcles_ref(
                X, k,
                d=self.d,
                maxIters=self.maxIters,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                epsilon=self.epsilon,
                maxItersForKmeans=self.maxItersForKmeans,
                nInitForKmeans=self.nInitForKmeans,
                verbose=verbose,
                path_ref_dir=self.path_ref_dir
            )
        else:
            result = mcles_sc(
                X, k,
                d=self.d,
                maxIters=self.maxIters,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                epsilon=self.epsilon,
                maxItersForKmeans=self.maxItersForKmeans,
                nInitForKmeans=self.nInitForKmeans,
                verbose=verbose
            )

        self.W = result['W']
        self.H = result['H']
        self.S = result['S']
        self.P = result['P']
        self.objs = result['objs']
