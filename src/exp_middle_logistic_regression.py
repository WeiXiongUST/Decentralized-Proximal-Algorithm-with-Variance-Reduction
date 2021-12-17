import numpy as np
from misc.get_data import load_generated_data
from misc.gen_matrix import ring_graph2
from misc.problems import LogisticRegression
from misc.miscs import PARAS
from run_exp import run_single_exp
from optimizer import NIDS, PGEXTRA, PMGT_LSVRG_FastMix, PMGT_SAGA_FastMix,Prox_ED

para1 = PARAS(num_agent=20, n=6400, dim=64, lamb1=1/(20*6400), lamb2=6400*1e-5, n_iter=500, tau=250, batch_size=1024,
              stepsize_LSVRG=1.15, stepsize_SAGA=0.45, stepsize_NIDS=0.4, stepsize_ED=0.4, nmix_LSVRG=10, nmix_SAGA=5)

para2 = PARAS(num_agent=20, n=6400, dim=64, lamb1=1/(20*6400), lamb2=6400*1e-7, n_iter=700, tau=250, batch_size=1024,
              stepsize_LSVRG=6.1, stepsize_SAGA=4.0, stepsize_NIDS=1.05, stepsize_ED=1.1, nmix_LSVRG=15, nmix_SAGA=10)
paras = [para1, para2]

X, Y = load_generated_data(20*6400)
W, alpha = ring_graph2(20, 0.2)

for run in range(len(paras)):
    if run == 0:
        continue
    print('exp with m = {}, n = {}, alpha = {}'.format(paras[run].num_agent, paras[run].n, alpha))
    problem = LogisticRegression(paras[run].num_agent, paras[run].n, paras[run].dim, sigma=paras[run].lamb1,
                                 lamb=paras[run].lamb2, X=X, Y=Y)
    x_0 = np.random.rand(paras[run].dim, paras[run].num_agent)  # initialization
    optimizers = [
        PMGT_LSVRG_FastMix(problem, paras[run].n_iter, x_0, paras[run].stepsize_LSVRG, W, paras[run].nmix_LSVRG,
                           paras[run].batch_size, prob=1/paras[run].n*paras[run].batch_size, verbose=False),
        PMGT_SAGA_FastMix(problem, paras[run].n_iter, x_0, paras[run].stepsize_SAGA, W, paras[run].nmix_SAGA,
                          paras[run].batch_size, verbose=False),
        PGEXTRA(problem, W, x_0, paras[run].n_iter, stepsize=paras[run].stepsize_NIDS),
        NIDS(problem, W, x_0, paras[run].n_iter, stepsize=paras[run].stepsize_NIDS),
        Prox_ED(problem, W, x_0, paras[run].n_iter, stepsize=paras[run].stepsize_ED)
    ]
    run_single_exp(optimizers, name='kappa02_' + str(paras[run].lamb2 / 6400)[-2:])
