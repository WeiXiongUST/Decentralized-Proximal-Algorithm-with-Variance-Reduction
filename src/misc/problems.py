import numpy as np


class Problem(object):
    """
    The problem is responsible for splitting data (equally) and provides two oracles:
    1. evaluation of f(x) and 2. evaluation of \nabla f(x), \nabla f_i(x), and \nabla f_{i,j}(x)
    """
    def __init__(self, num_agent, num_data, dim, sigma, lamb, X=None, Y=None, **kwargs):
        self.num_agent = num_agent # Number of agents
        self.data_total = self.num_agent * num_data # Number of the whole data set
        self.dim = dim  # Dimension of the variable
        self.data_sizes = np.ones(self.num_agent, dtype=int) * num_data
        self.data_size = num_data
        self.lamb = lamb # coefficient of the l2 regularization term
        self.sigma = sigma # coefficient of the l1 regularization term

        norm = np.sqrt(np.linalg.norm(X.T.dot(X), 2) / self.data_total)
        self.X_total = X / (norm + lamb)
        #self.X_total = X
        self.Y_total = Y

        self.X = self.split_data(self.data_sizes, self.X_total)
        self.Y = self.split_data(self.data_sizes, self.Y_total)

    def split_data(self, m, X):
        # Helper function to split data according to the number of training samples per agent.
        cumsum = m.cumsum().astype(int).tolist()
        inds = zip([0] + cumsum[:-1], cumsum)
        return [X[start:end] for (start, end) in inds ] # Return the reference of data, which is contiguous

    def grad(self, w, i=None, j=None):
        pass

    def f(self, w, i=None, j=None):
        pass

    def gradient_check(self):
        # Check the gradient implementation by numerical gradient.
        w = np.random.rand(self.dim)
        delta = np.zeros(self.dim)
        grad = np.zeros(self.dim)
        eps = 1e-6

        for i in range(self.dim):
            delta[i] = eps
            grad[i] = (self.f(w+delta) - self.f(w-delta)) / 2 / eps
            delta[i] = 0

        if np.linalg.norm(grad - self.grad(w)) > eps:
            print('Grad check failed!')
            return False
        else:
            print('Grad check succeeded!')
            return True


class LogisticRegression(Problem):
    def __init__(self, num_agent, num_data, dim, sigma, lamb, X=None, Y=None, **kwargs):
        super().__init__(num_agent, num_data, dim, sigma, lamb, X, Y)

    def _logit(self, X, w):
        #print(-X.dot(w))
        return 1 / (1 + np.exp(-X.dot(w)))

    def grad(self, w, i=None, j=None):
        if i is None: # Return \nabla f
            return self.X_total.T.dot(self._logit(self.X_total, w) - self.Y_total) / self.data_total + w * self.lamb
        elif j is None: # Return \nabla f_i
                return self.X[i].T.dot(self._logit(self.X[i], w) - self.Y[i]) / self.data_sizes[i] + w * self.lamb
        else: # Return \nabla f_{ij}
            if type(j) is np.ndarray:
                return (self._logit(self.X[i][j], w) - self.Y[i][j]).dot(self.X[i][j]) / len(j) + w * self.lamb
            else:
                return (self._logit(self.X[i][j], w) - self.Y[i][j]) * self.X[i][j] + w * self.lamb

    def f(self, w, i=None, j=None):
        if i is None:  # Return f(x)
            tmp = self.X_total.dot(w)
            return - np.sum(
                    (self.Y_total) * tmp - np.log(1 + np.exp(tmp))
                    ) / self.data_total + np.sum(w**2) * self.lamb / 2 + self.sigma * np.linalg.norm(w, ord=1)
        elif j is None:  # Return f_i(w)
            tmp = self.X[i].dot(w)
            return - np.sum(
                    (self.Y[i]) * tmp - np.log(1 + np.exp(tmp))
                    ) / self.data_size + self.sigma * np.linalg.norm(w, ord=1) + np.sum(w**2) * self.lamb / 2
        else:  # Return f_{ij}(w)
            tmp = self.X[i][j].dot(w)
            return -((self.Y[i][j]) * tmp - np.log(1 + np.exp(tmp))) + self.lamb * np.linalg.norm(w, ord=1) + np.sum(w**2) * self.lamb / 2


