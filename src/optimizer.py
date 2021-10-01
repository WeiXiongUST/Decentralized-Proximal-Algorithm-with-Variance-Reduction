import numpy as np
eps = 1e-10


class DecentralizedOptimizer(object):
    def __init__(self, problem, num_iters=100, x_0=None, stepsize=0.1, W=None, verbose=False, name=None):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.problem = problem
        self.num_iters = num_iters
        self.x_0 = x_0
        self.stepsize = stepsize
        self.W = W
        self.verbose = verbose
        self.num_agent = self.problem.num_agent

        self.t = 0
        self.x = self.x_0.copy()

        self.n_grad = np.zeros(self.num_iters)
        self.n_comm = np.zeros(self.num_iters)

        self.func_error = np.zeros(self.num_iters)

    def get_name(self):
        return self.name

    def f(self, w, i=None, j=None):
        return self.problem.f(w, i, j)

    def grad(self, w, i=None, j=None):
        """This function returns the gradients and counts the number of component gradient evaluations."""
        if i is None:  # The full gradient
            self.n_grad[self.t] += self.problem.data_total
        elif j is None:  # samples at agent i
            self.n_grad[self.t] += self.problem.data_sizes[i]
        else:
            if type(j) is np.ndarray:  # a mini-batch
                self.n_grad[self.t] += len(j)
            else:
                self.n_grad[self.t] += 1  # 1 sample

        return self.problem.grad(w, i, j)

    def prox_plus(self, X):
        """Operator: max(X, 0)"""
        below = X < 0
        X[below] = 0
        return X

    def prox_l1(self, X, lamb):
        """Proximal operator for the l1 regularization"""
        X = self.prox_plus(np.abs(X) - lamb) * np.sign(X)
        return X

    def save_metric(self):
        self.func_error[self.t] = self.f(self.x.mean(axis=1))

    def get_results(self, tau=250):
        self.n_cost = np.zeros(self.num_iters+1)

        for i in range(self.t):
            self.n_cost[i] = self.n_comm[i] * tau + self.n_grad[i] / self.problem.num_agent

        result = {
                'func_error': self.func_error[:self.t],
                'n_grad': self.n_grad[:self.t] / self.problem.num_agent / 10000,
                'n_comm': self.n_comm[:self.t],
                'n_cost': self.n_cost[:self.t] / 10000,
                't': self.t,
                'name': self.name
                }
        return result

    def init_iter(self):
        self.n_comm[self.t] += self.n_comm[self.t-1]
        self.n_grad[self.t] += self.n_grad[self.t-1]

    def optimize(self):
        for self.t in range(0, self.num_iters):
            self.init_iter()
            self.update()
            self.save_metric()
        return self.get_results()

    def update(self):
        pass

class PMGT_SGD(DecentralizedOptimizer):
    def __init__(self, problem, num_iters=100, x_0=None, stepsize=0.1, W=None, K=1, batch_size=1, verbose=False, name=None):
        super().__init__(problem, num_iters=num_iters, x_0=x_0, stepsize=stepsize, W=W, verbose=verbose, name=name)
        self.batch_size = batch_size
        self.s = np.zeros((self.problem.dim, self.problem.num_agent))
        self.v_last = np.zeros((self.problem.dim, self.problem.num_agent))
        self.K = K
        self.W = np.linalg.matrix_power(W, self.K)

        self.x = self.x_0.copy()
        self.w = self.x_0.copy()
        self.lamb1 = problem.sigma

        for i in range(self.num_agent):
            self.s[:, i] = self.grad(self.x[:, i], i)
            self.v_last[:, i] = self.problem.grad(self.x[:, i], i)

    def update(self):
        for i in range(self.num_agent):
            tmp = self.x[:, i] - self.stepsize * self.s[:, i]
            self.x[:, i] = self.prox_l1(tmp, self.lamb1*self.stepsize)
        self.x = self.x.dot(self.W)
        self.n_comm[self.t] += self.K

        for i in range(self.num_agent):
            ji = np.random.randint(0, self.problem.data_sizes[i], self.batch_size)
            v = self.grad(self.x[:, i], i, ji)
            self.s[:, i] += v - self.v_last[:, i]
            self.v_last[:, i] = v

        self.s = self.s.dot(self.W)
        self.n_comm[self.t] += self.K


class PMGT_LSVRG(DecentralizedOptimizer):
    def __init__(self, problem, num_iters=100, x_0=None, stepsize=0.1, W=None, K=1, batch_size=1, prob=0.1, verbose=False, name=None):
        super().__init__(problem, num_iters=num_iters, x_0=x_0, stepsize=stepsize, W=W, verbose=verbose, name=name)
        self.batch_size = batch_size
        self.prob = prob
        self.full_gradient = np.zeros((self.problem.dim, self.problem.num_agent))
        self.s = np.zeros((self.problem.dim, self.problem.num_agent))
        self.v_last = np.zeros((self.problem.dim, self.problem.num_agent))
        ### not fast mix here
        self.K = K
        self.W = np.linalg.matrix_power(W, self.K)

        self.x = self.x_0.copy()
        self.w = self.x_0.copy()
        self.lamb1 = problem.sigma

        for i in range(self.num_agent):
            self.full_gradient[:, i] = self.grad(self.x[:, i], i)
            self.s[:, i] = self.grad(self.x[:, i], i)
            self.v_last[:, i] = self.problem.grad(self.x[:, i], i)

    def update(self):
        for i in range(self.num_agent):
            tmp = self.x[:, i] - self.stepsize * self.s[:, i]
            self.x[:, i] = self.prox_l1(tmp, self.lamb1*self.stepsize)
        self.x = self.x.dot(self.W)
        self.n_comm[self.t] += self.K

        for i in range(self.num_agent):
            ji = np.random.randint(0, self.problem.data_sizes[i], self.batch_size)
            v = self.grad(self.x[:, i], i, ji) - self.grad(self.w[:, i], i, ji) + self.full_gradient[:, i]
            self.s[:, i] += v - self.v_last[:, i]
            self.v_last[:, i] = v
            if np.random.random() < self.prob:
                self.w[:, i] = self.x[:, i]
                self.full_gradient[:, i] = self.grad(self.x[:, i], i)

        self.s = self.s.dot(self.W)
        self.n_comm[self.t] += self.K


class PMGT_SAGA(DecentralizedOptimizer):
    def __init__(self, problem, num_iters=100, x_0=None, stepsize=0.1, W=None, K=1, batch_size=1, verbose=False, name=None):
        super().__init__(problem, num_iters=num_iters, x_0=x_0, stepsize=stepsize, W=W, verbose=verbose, name=name)
        self.batch_size = batch_size
        self.gradient_table = np.zeros((self.problem.num_agent, self.problem.dim, self.problem.data_sizes[0]))
        self.s = np.zeros((self.problem.dim, self.problem.num_agent))
        self.v_last = np.zeros((self.problem.dim, self.problem.num_agent))
        self.K = K
        self.W = np.linalg.matrix_power(W, self.K)

        self.x = self.x_0.copy()
        self.w = self.x_0.copy()
        self.lamb1 = problem.sigma

        for i in range(self.problem.num_agent):
            self.s[:, i] = self.grad(self.x[:, i], i)
            self.v_last[:, i] = self.problem.grad(self.x[:, i], i)

        for i in range(self.problem.num_agent):
            for j in range(self.problem.data_sizes[i]):
                self.gradient_table[i, :, j] = self.grad(self.x[:,i], i, j)

    def update(self):
        for i in range(self.problem.num_agent):
            tmp = self.x[:, i] - self.stepsize * self.s[:, i]
            self.x[:, i] = self.prox_l1(tmp, self.stepsize*self.lamb1)

        self.x = self.x.dot(self.W)
        self.n_comm[self.t] += self.K

        for i in range(self.problem.num_agent):
            randIndex = np.random.randint(0, self.problem.data_sizes[i], self.batch_size)
            temp_gradient = np.zeros(self.problem.dim)
            v = -np.sum(self.gradient_table[i, :, randIndex], axis=0) / self.batch_size + np.sum(self.gradient_table[i, :,:], axis=1) / self.problem.data_sizes[i]
            for ji in randIndex:
                self.gradient_table[i, :, ji] = self.grad(self.x[:, i], i, ji)
                temp_gradient += self.gradient_table[i, :, ji]
            temp_gradient /= self.batch_size

            v += temp_gradient
            self.s[:, i] += v.reshape(np.shape(self.v_last[:,i])) - self.v_last[:, i]
            self.v_last[:, i] = v

        self.s = self.s.dot(self.W)
        self.n_comm[self.t] += self.K


class PGEXTRA(DecentralizedOptimizer):
    def __init__(self, problem, W, x_0, num_iters=100, stepsize=0.1, name="NIDS"):
        super().__init__(problem, num_iters=num_iters, x_0=x_0, stepsize=stepsize, W=W, name=name)
        self.grad_last = None
        self.z = self.x_0.copy()
        self.lamb = problem.sigma

        self.W = W
        tmp = 0.5
        self.W_s = self.W*tmp + np.eye(self.problem.num_agent) * (1 - tmp)

    def update(self):
        self.n_comm[self.t] += 1

        if self.t == 0:
            self.grad_last = np.zeros((self.problem.dim, self.problem.num_agent))

            for i in range(self.problem.num_agent):
                self.grad_last[:, i] = self.grad(self.x[:, i], i)
            self.z = self.x.dot(self.W) - self.stepsize * self.grad_last
        else:
            tmp = self.x.dot(self.W) + self.z - self.x_last.dot(self.W_s)
            tmp += self.stepsize * self.grad_last
            for i in range(self.problem.num_agent):
                self.grad_last[:, i] = self.grad(self.x[:, i], i)
            tmp -= self.stepsize * self.grad_last
            self.z = tmp

        tmp_x = np.zeros((self.problem.dim, self.problem.num_agent))
        for i in range(self.problem.num_agent):
            tmp_x[:, i] = self.prox_l1(self.z[:, i], self.stepsize * self.lamb)

        # Update variables
        self.x, self.x_last = tmp_x, self.x

class NIDS(DecentralizedOptimizer):

    def __init__(self, problem, W, x_0, num_iters=100, stepsize=0.1, name="PG-EXTRA"):
        super().__init__(problem, num_iters=num_iters, x_0=x_0, stepsize=stepsize, W=W, name=name)
        self.grad_last = None
        self.z = self.x_0.copy()
        self.lamb = problem.sigma

        self.W = W
        tmp = 0.5
        self.W_s = self.W*tmp + np.eye(self.problem.num_agent) * (1 - tmp)

    def update(self):
        self.n_comm[self.t] += 1

        if self.t == 0:
            self.grad_last = np.zeros((self.problem.dim, self.problem.num_agent))

            for i in range(self.problem.num_agent):
                self.grad_last[:, i] = self.grad(self.x[:, i], i)
            self.z -= self.stepsize * self.grad_last

        else:
            tmp = self.x.dot(self.W) + self.x - self.x_last.dot(self.W_s)
            v = self.stepsize * self.grad_last
            for i in range(self.problem.num_agent):
                self.grad_last[:, i] = self.grad(self.x[:, i], i)
            v -= self.stepsize * self.grad_last
            tmp = tmp + v.dot(self.W_s)
            self.z += tmp - self.x

        tmp_x = np.zeros((self.problem.dim, self.problem.num_agent))
        for i in range(self.problem.num_agent):
            tmp_x[:, i] = self.prox_l1(self.z[:, i], self.stepsize * self.lamb)

        # Update variables
        self.x, self.x_last = tmp_x, self.x
        self.s = self.grad_last
