import itertools

import numpy as np
from matplotlib import pyplot as plt

SIZE_XY = 15
SIZE_LEGEND = 18
MAKER = ['*', 'x', '+', 'o']
COLOR = ['b', 'c', 'k', 'r', 'k']
GAP = 20

def LINE_STYLES():
    return itertools.cycle([
        '-k*', '-cx', '--b<', '--ro', '--mP'
    ])


def auto_plot_results(result_sequence, pathid=None, title='11'):
    """This function plots the sub-optimality with respect to gradient evaluation, communication and cost.
    The figures will be stored but will not be shown. This is for hyperparameter searching.
    """
    if pathid is not None:
        path = "./fig/" + pathid
    else:
        path = "./fig/xx"

    min_value = np.min([np.min(result_sequence[i]['func_error']) for i in range(len(result_sequence))])
    legends = []
    grad = []
    comm = []
    cost = []
    for result in result_sequence:
        zz = result['func_error'] - min_value
        if len(np.where(zz < 1e-12)[0]) > 0:
            index = np.where(zz < 1e-12)[0][0]
            grad.append(result['n_grad'][index])
            comm.append(result['n_comm'][index])
            cost.append(result['n_cost'][index])
        else:
            grad.append(np.max(result['n_grad']))
            comm.append(np.max(result['n_comm']))
            cost.append(np.max(result['n_cost']))
    plt.figure()
    plt.subplot(1, 3, 1)
    for result in result_sequence:
        plt.semilogy(result['n_grad'], result['func_error'] - min_value + 1e-15)
        legends.append(result['name'])

    plt.ylabel(r"$\log_{10}[h({\bar{\mathbf{x}}}^{(t)}) - h({\mathbf{x}}^\star)]$")
    plt.xlabel('#Grad Evaluations(x1e4)')
    xmax = np.max(grad) * 1.2
    #plt.xlim(0, xmax)
    plt.title(title)
    plt.grid()
    plt.legend(legends)

    with open("./fig/record.txt", 'a+') as f:
        f.write(title + '\t' + str([[legends[i], grad[i], comm[i], cost[i]] for i in range(len(legends))]) + '\n')

    plt.subplot(1, 3, 2)
    for result in result_sequence:
        plt.semilogy(result['n_comm'], result['func_error'] - min_value + 1e-15)
    plt.ylabel(r"$\log_{10}[h({\bar{\mathbf{x}}}^{(t)}) - h({\mathbf{x}}^\star)]$")
    plt.xlabel('#Communications')
    xmax = np.max(comm) * 1.2
    #plt.xlim(0, xmax)
    plt.grid()
    plt.legend(legends)

    plt.subplot(1, 3, 3)
    for result in result_sequence:
        plt.semilogy(result['n_cost'], result['func_error'] - min_value + 1e-15)
    plt.ylabel(r"$\log_{10}[h({\bar{\mathbf{x}}}^{(t)}) - h({\mathbf{x}}^\star)]$")
    plt.xlabel('#Costs(x1e4)')
    xmax = np.max(cost) * 1.2
    #plt.xlim(0, xmax)
    plt.grid()
    plt.legend(legends)

    plt.savefig(path)
    #plt.show()
    plt.close()


def plot_results(result_sequence):
    """This function plots the sub-optimality with respect to gradient evaluation, communication and cost
    without storing them."""
    min_value = np.min([np.min(result_sequence[i]['func_error']) for i in range(len(result_sequence))])
    legends = []

    plt.figure()
    for result in result_sequence:
        plt.semilogy(result['n_grad'], result['func_error'] - min_value + 1e-15)
        legends.append(result['name'])
    plt.ylabel(r"$\log_{10}[h({\bar{\mathbf{x}}}^{(t)}) - h({\mathbf{x}}^\star)]$")
    plt.xlabel('#Grad Evaluations(x1e4)')
    plt.grid()
    plt.legend(legends)

    plt.figure()
    for result in result_sequence:
        plt.semilogy(result['n_comm'], result['func_error'] - min_value + 1e-15)
    plt.ylabel(r"$\log_{10}[h({\bar{\mathbf{x}}}^{(t)}) - h({\mathbf{x}}^\star)]$")
    plt.xlabel('#Communications')
    plt.grid()
    plt.legend(legends)

    plt.figure()
    for result in result_sequence:
        plt.semilogy(result['n_cost'], result['func_error'] - min_value + 1e-15)
    plt.ylabel(r"$\log_{10}[h({\bar{\mathbf{x}}}^{(t)}) - h({\mathbf{x}}^\star)]$")
    plt.xlabel('#Costs(x1e4)')
    plt.grid()
    plt.legend(legends)

    plt.show()


def final_plot_results(result_sequence, path_storage):
    """This function plots the sub-optimality with respect to gradient evaluation, communication and cost.
    This is for the figures used in the paper.
    """
    if path_storage is not None:
        path = "./results/final_fig/" + path_storage
    else:
        path = "./results/final_fig/xx"
    if path_storage[0] == 'e':
        legend_comm = True
    else:
        legend_comm = False

    min_value = np.min([np.min(result_sequence[i]['func_error']) for i in range(len(result_sequence))])
    legends = []
    grad = []
    comm = []
    cost = []
    idx = []

    for result in result_sequence:
        zz = result['func_error'] - min_value
        if len(np.where(zz < 1e-12)[0]) > 0:
            index = np.where(zz < 1e-12)[0][0]
            grad.append(result['n_grad'][index])
            comm.append(result['n_comm'][index])
            cost.append(result['n_cost'][index])
            idx.append(index)
        else:
            grad.append(np.max(result['n_grad']))
            comm.append(np.max(result['n_comm']))
            cost.append(np.max(result['n_cost']))
    plt.figure()
    line_style = LINE_STYLES()
    t = 0
    for result in result_sequence:
        index = idx[t]
        t += 1
        plt.semilogy(result['n_grad'], result['func_error'] - min_value + 1e-15, line_style.__next__(), markevery=int(index/GAP))
        if result['name'] == 'PMGT_LSVRG_FastMix':
            legends.append('PMGT_LSVRG')
        elif result['name'] == 'PMGT_SAGA_FastMix':
            legends.append('PMGT_SAGA')
        else:
            legends.append(result['name'])
    plt.ylabel(r"$\log_{10}[h({\bar{\mathbf{x}}}^{t}) - h({\mathbf{x}}^\star)]$", size=SIZE_XY)
    plt.xlabel('#Grad Evaluations (x1e4)', size=SIZE_XY)

    tmp = np.argmax(grad)
    grad[tmp] = -1
    xmax = np.max(grad) * 1.5

    plt.xlim(0, xmax)
    plt.ylim(1e-13, 1)

    plt.grid()
    plt.tight_layout()
    plt.tick_params(labelsize=SIZE_XY)

    plt.savefig(path + 'grad')
    plt.close()

    plt.figure()
    line_style = LINE_STYLES()
    t = 0
    for result in result_sequence:
        index = idx[t]
        t += 1
        plt.semilogy(result['n_comm'], result['func_error'] - min_value + 1e-15, line_style.__next__(), markevery=int(index/GAP))
    plt.ylabel(r"$\log_{10}[h({\bar{\mathbf{x}}}^{t}) - h({\mathbf{x}}^\star)]$", size=SIZE_XY)
    plt.xlabel('#Communications', size=SIZE_XY)

    tmp = np.argmax(comm)
    comm[tmp] = -1
    xmax = np.max(comm) * 1.5

    plt.xlim(0, xmax)
    plt.ylim(1e-13, 1)
    if legend_comm:
        plt.legend(legends, fontsize=SIZE_LEGEND, loc=5)

    plt.grid()
    plt.tight_layout()
    plt.tick_params(labelsize=SIZE_XY)

    plt.savefig(path + 'comm')
    plt.close()

    plt.figure()
    t = 0
    line_style = LINE_STYLES()
    for result in result_sequence:
        index = idx[t]
        t += 1
        plt.semilogy(result['n_cost'], result['func_error'] - min_value + 1e-15, line_style.__next__(), markevery=int(index/GAP))
    plt.ylabel(r"$\log_{10}[h({\bar{\mathbf{x}}}^{t}) - h({\mathbf{x}}^\star)]$", size=SIZE_XY)
    plt.xlabel('#Costs (x1e4)', size=SIZE_XY)

    tmp = np.argmax(cost)
    cost[tmp] = -1
    xmax = np.max(cost) * 1.5

    plt.xlim(0, xmax)
    plt.ylim(1e-13, 1)
    plt.grid()
    plt.legend(legends, fontsize=SIZE_LEGEND, loc=5)
    plt.tick_params(labelsize=SIZE_XY)
    plt.tight_layout()
    plt.savefig(path + 'cost')
    plt.close()
