import numpy as np
from misc.get_data import load_generated_data
from misc.gen_matrix import ring_graph2
from misc.problems import LogisticRegression
from misc.miscs import PARAS
from run_exp import run_single_exp
from optimizer import NIDS, PGEXTRA, PMGT_LSVRG_FastMix, PMGT_SAGA_FastMix


para1 = PARAS(num_agent=20, n=6400, dim=64, lamb1=1/(20*6400), lamb2=6400*1e-7, n_iter=1200, tau=250, batch_size=1024,
              stepsize_LSVRG=0.8, stepsize_SAGA=0.9, stepsize_NIDS=2.9, nmix_LSVRG=1, nmix_SAGA=1)

para2 = PARAS(num_agent=20, n=6400, dim=64, lamb1=1/(20*6400), lamb2=6400*1e-7, n_iter=300, tau=250, batch_size=1024,
              stepsize_LSVRG=3.4, stepsize_SAGA=3.4, stepsize_NIDS=2.9, nmix_LSVRG=3, nmix_SAGA=3)

para3 = PARAS(num_agent=20, n=6400, dim=64, lamb1=1/(20*6400), lamb2=6400*1e-7, n_iter=300, tau=250, batch_size=1024,
              stepsize_LSVRG=5.9, stepsize_SAGA=5.7, stepsize_NIDS=2.9, nmix_LSVRG=5, nmix_SAGA=5)

para4 = PARAS(num_agent=20, n=6400, dim=64, lamb1=1/(20*6400), lamb2=6400*1e-7, n_iter=300, tau=250, batch_size=1024,
              stepsize_LSVRG=5.9, stepsize_SAGA=5.7, stepsize_NIDS=2.9, nmix_LSVRG=10, nmix_SAGA=10)

para5 = PARAS(num_agent=20, n=6400, dim=64, lamb1=1/(20*6400), lamb2=6400*1e-7, n_iter=300, tau=250, batch_size=1024,
              stepsize_LSVRG=5.9, stepsize_SAGA=5.7, stepsize_NIDS=2.9, nmix_LSVRG=30, nmix_SAGA=30)

paras = [para1, para2, para3, para4, para5]

X, Y = load_generated_data(20*6400)
W, alpha = ring_graph2(20, 1)

optimizers = []
x_0 = np.random.rand(paras[0].dim, paras[0].num_agent)  # initialization
problem = LogisticRegression(paras[0].num_agent, paras[0].n, paras[0].dim, sigma=paras[0].lamb1,
                             lamb=paras[0].lamb2, X=X, Y=Y)
print('exp with m = {}, n = {}, alpha = {}'.format(paras[0].num_agent, paras[0].n, alpha))

for run in range(len(paras)):
    optimizers.append(PMGT_LSVRG_FastMix(problem, paras[run].n_iter, x_0, paras[run].stepsize_LSVRG, W, paras[run].nmix_LSVRG,
                           paras[run].batch_size, prob=1/paras[run].n*paras[run].batch_size, verbose=False))
    optimizers[-1].name = "K=" + str(paras[run].nmix_LSVRG)

run_single_exp(optimizers, name='exp_kappa1_e7_nmix')

