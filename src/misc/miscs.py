import numpy as np


class PARAS(object):
    def __init__(self, num_agent, n, dim, lamb1, lamb2, n_iter, tau, batch_size,
                 stepsize_LSVRG, stepsize_SAGA, stepsize_NIDS, stepsize_ED, nmix_LSVRG, nmix_SAGA):
        self.num_agent = num_agent
        self.n = n
        self.dim = dim
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.n_iter = n_iter
        self.tau = tau
        self.batch_size = batch_size
        self.stepsize_LSVRG = stepsize_LSVRG
        self.stepsize_SAGA = stepsize_SAGA
        self.stepsize_NIDS = stepsize_NIDS
        self.stepsize_ED = stepsize_ED
        self.nmix_LSVRG = nmix_LSVRG
        self.nmix_SAGA = nmix_SAGA
