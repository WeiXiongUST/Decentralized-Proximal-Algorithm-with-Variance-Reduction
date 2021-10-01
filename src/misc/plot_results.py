import numpy as np
from matplotlib import pyplot as plt

def plot_results(result_sequence):
    """This function plots the sub-optimality with respect to gradient evaluation, communication and cost."""
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
