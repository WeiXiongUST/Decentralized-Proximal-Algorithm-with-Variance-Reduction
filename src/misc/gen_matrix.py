import numpy as np
import networkx as nx


def get_matrix(p, m):
    """
    This function generates a gossip matrix of size m x m where
    1. Each pair of agents are connected with probability p;
    2. return: gossip matrix W, 1 - 2nd_largest_singular_value.
    """
    G = nx.erdos_renyi_graph(m, p)

    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(m, p)

    L = nx.laplacian_matrix(G).todense()
    val, vec = np.linalg.eig(L)
    lamb1 = np.max(val)
    W = np.eye(m) - L / lamb1

    val, vec = np.linalg.eig(W)
    val[0] = -777
    mixing_rate = np.max(val)
    W = np.array(W)
    return W, 1 - mixing_rate





