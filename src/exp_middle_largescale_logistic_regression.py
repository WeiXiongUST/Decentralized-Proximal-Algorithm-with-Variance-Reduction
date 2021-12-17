import numpy as np
from misc.get_data import load_generated_data
from misc.gen_matrix import ring_graph2
from misc.problems import LogisticRegression
from misc.miscs import PARAS
from run_exp import run_single_exp
from optimizer import NIDS, PGEXTRA, PMGT_LSVRG_FastMix, PMGT_SAGA_FastMix, Prox_ED


para1 = PARAS(num_agent=20, n=64000, dim=64, lamb1=1/(20*64000), lamb2=64000*1e-6, n_iter=300, tau=250, batch_size=8192,
              stepsize_LSVRG=1, stepsize_SAGA=0.5, stepsize_NIDS=0.45, stepsize_ED=0.2, nmix_LSVRG=10, nmix_SAGA=5)

para2 = PARAS(num_agent=20, n=64000, dim=64, lamb1=1/(20*64000), lamb2=64000*1e-8, n_iter=450, tau=250, batch_size=8192,
              stepsize_LSVRG=5.8, stepsize_SAGA=4, stepsize_NIDS=1.4, stepsize_ED=1.1, nmix_LSVRG=15, nmix_SAGA=10)
paras = [para1, para2]

X, Y = load_generated_data(20*64000)
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
        Prox_ED(problem, W, x_0, paras[run].n_iter*4, stepsize=paras[run].stepsize_ED)
    ]
    run_single_exp(optimizers, name='kappa02_largescale_' + str(paras[run].lamb2 / 64000)[-2:])
