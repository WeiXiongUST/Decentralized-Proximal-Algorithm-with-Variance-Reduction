import numpy as np
from misc.get_data import load_a9a
from misc.gen_matrix import get_matrix
from misc.problems import LogisticRegression
from misc.plot_results import plot_results
from optimizer import PMGT_LSVRG, PMGT_SAGA, NIDS, PMGT_SGD, PGEXTRA

# Parameters
X, Y = load_a9a()
num_agent = 20
n = 1625  # size of local dataset
dim = 123

lamb1 = 1 / (n) # L1 regularization
lamb2 = n*1e-5  # L2 regularization

n_iters = 400
tau = 250  # cost of communication / cost of gradient evaluation = tau

stepsize_LSVRG = 5.5
stepsize_SAGA = 3.8
stepsize_NIDS = 5
batch_size = 128
n_mix = 1

problem = LogisticRegression(num_agent, n, dim, sigma=lamb1, lamb=lamb2, X=X, Y=Y)
x_0 = np.random.rand(dim, num_agent)  # initialization

W, alpha = get_matrix(0.9, num_agent)
while(alpha < 0.8 or alpha > 0.81):
    W, alpha = get_matrix(0.9, num_agent)
    alpha = abs(alpha)

optimizers = [
    PMGT_LSVRG(problem, n_iters, x_0, stepsize_NIDS, W, n_mix, batch_size, prob=1/n*batch_size, verbose=False),
    PMGT_SAGA(problem, n_iters, x_0, stepsize_NIDS, W, n_mix, batch_size, verbose=False),
    PGEXTRA(problem, W, x_0, n_iters, stepsize=stepsize_LSVRG),
    NIDS(problem, W, x_0, n_iters, stepsize=stepsize_SAGA),
]

for i in range(len(optimizers)):
    optimizers[i].optimize()

result_sequence = [optimizers[i].get_results(tau) for i in range(len(optimizers))]
plot_results(result_sequence)



