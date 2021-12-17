import numpy as np
import networkx as nx


def get_matrix(p, m):
    """
    This function generates a gossip matrix of size m x m where
    1. Each pair of agents are connected with probability p;
    2. return: gossip matrix W, 2nd_largest_singular_value.
    """
    G = nx.erdos_renyi_graph(m, p)

    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(m, p)

    L = nx.laplacian_matrix(G).todense()
    val, vec = np.linalg.eig(L)
    lamb1 = np.max(val)
    W = np.eye(m) - L / lamb1

    W = np.array(W)
    a, b, c = np.linalg.svd(W)
    return W, b[1]


def ring_graph2(n, kappa, k=1):
    """
    This function returns a gossip matrix W generated as follows:
    1 We get a matrix N from the ring graph with n nodes where each agent is connected with two neighbors;
    2 We return W = kappa * N + (1 - kappa) * N;
    3 For small kappa, the gap will also be small corresponding to a low consensus rate.
    """
    G = nx.Graph()
    sources = np.arange(n)
    for i in range(1, k + 1):
        targets = np.roll(sources, i)
        G.add_edges_from(zip(sources, targets))
    L = nx.laplacian_matrix(G).todense()
    val, vec = np.linalg.eig(L)
    lamb1 = np.max(val)
    W = np.eye(n) - L / lamb1
    W = np.array(W)
    W = kappa * W + (1 - kappa) * np.eye(n)
    a, b, c = np.linalg.svd(W)
    return W, b[1]


'''
for i in range(8, 50):
    zz, a = ring_graph2(i)
    print(i, a)

zz, a = ring_graph2(20, 1)
for kappa in [1, 0.2, 0.1, 0.02]:
    W = kappa * zz + (1-kappa) * np.eye(20)
    a, b, c = np.linalg.svd(W)
    print(b[1])
'''
