import numpy as np
from misc.get_data import load_a9a, load_cifar10
from misc.gen_matrix import get_matrix
from misc.problems import LogisticRegression
from misc.plot_results import auto_plot_results, plot_results, final_plot_results
from optimizer import PMGT_LSVRG, PMGT_SAGA, NIDS, Prox_ED, PGEXTRA, PMGT_SAGA_FastMix, PMGT_LSVRG_FastMix, prox_ATC
import copy
import time


def run_single_exp(optimizers, name='xx', save=True):
    for i in range(len(optimizers)):
        start = time.perf_counter()
        optimizers[i].optimize()
        print(optimizers[i].get_name() + ' is simulated with ', time.perf_counter() - start)

    result_sequence = [optimizers[i].get_results(tau=250) for i in range(len(optimizers))]
    final_plot_results(result_sequence, path_storage=name)
    if save:
        np.save('results/' + name +'.npy', result_sequence)


def run_single_exp_parasearch(optimizers, name='xx', title='111'):
    for i in range(len(optimizers)):
        optimizers[i].optimize()
    result_sequence = [optimizers[i].get_results(tau=250) for i in range(len(optimizers))]

    auto_plot_results(result_sequence, pathid=name, title=title)


def run_exps_exact(problem, W, x_0, n_iters, stepsizes, name='xx'):
    for i in range(len(stepsizes)):
        print(i, '...')
        optimizers = [
        Prox_ED(problem, W, x_0, n_iters, stepsize=stepsizes[i]),
        NIDS(problem, W, x_0, n_iters, stepsize=stepsizes[i]),
        ]
        run_single_exp_parasearch(optimizers, name=name + str(i), title=str(stepsizes[i]))


def run_exps_stochastic(problem, W, x_0, n_iters, stepsizes, batchsizes, n_mixs, stepsize_exact, name='xx'):
    for j in range(len(batchsizes)):
        for k in range(len(n_mixs)):
            for i in range(len(stepsizes)):
                optimizers = [
                PMGT_LSVRG_FastMix(problem, n_iters, x_0, stepsizes[i], W, n_mixs[k], batchsizes[j], prob=1/64000*batchsizes[j], verbose=False),
                PMGT_SAGA_FastMix(problem, n_iters, x_0, stepsizes[i], W, n_mixs[k], batchsizes[j], verbose=False),
                NIDS(problem, W, x_0, n_iters, stepsize=stepsizes[i]),
                ]
                start = time.perf_counter()
                run_single_exp_parasearch(optimizers, name=name + 'batchsize_' + str(batchsizes[j]) + '_K_' + str(n_mixs[k]) + '_' + str(i), title=str(stepsizes[i]) + '_'+ str(batchsizes[j]) + '_' + str(n_mixs[k]))
                print(time.perf_counter() - start)

