import numpy as np
from sklearn.datasets import fetch_rcv1
import scipy.sparse as sp


def load_a9a(path = 'data/a9a.txt'):
    """Load a9a data from path"""
    feature_size = 123
    num_of_data = 32500
    labels = np.zeros(num_of_data)
    rets = np.zeros(shape=[num_of_data, feature_size])
    i = 0
    for line in open(path, "r"):
        data = line.split(" ")
        label = int(float(data[0]))
        ids = []
        values = []
        for fea in data[1:]:
            if fea == '\n':
                continue
            id, value = fea.split(":")
            if int(id) > feature_size - 1:
                break
            ids.append(int(id))
            values.append(float(value))
        for (index, d) in zip(ids, values):
            rets[i, index] = d
        labels[i] = label
        i += 1
        if i >= num_of_data:
            break
    labels = (labels + 1 )/2
    return rets, labels


def load_cifar10(path='D:/TPAMI/TPAMI_Revision/data/cifar10/data_batch_'):
    train_data = np.zeros((12000, 3072))
    train_labels = np.zeros(12000)
    t = 0
    import pickle
    for i in range(1, 7):
        with open(path + str(i), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            for j in range(10000):
                if dict[b'labels'][j] == 0 or dict[b'labels'][j] == 1:
                    train_data[t] = dict[b'data'][j]
                    train_labels[t] = dict[b'labels'][j]
                    t += 1

    return train_data, train_labels


# CCAT, ECAT: 33, 59;
# negative: GCAT, MCAT: 70, 102

def load_rcv():
    X, y = fetch_rcv1(data_home='data/rcv1', return_X_y=True)
    tmp1 = []
    tmp3 = []
    N = 10000
    count1 = 0
    count2 = 0
    X = X[0:40000]
    X = X.todense()
    y = y.todense()
    data = np.zeros((N, 47236))
    t = 0
    zz = np.array(X[0])
    for i in range(X.shape[0]):
        target_vec = y[i]
        if (target_vec[0, 33] == 1 or target_vec[0, 59] == 1) and (target_vec[0, 70] == 1 or target_vec[0, 102] == 1):
            pass
        elif target_vec[0, 33] == 1 or target_vec[0, 59] == 1:
            if count1 < N / 2:
                zz = np.array(X[i])
                data[t] = zz[0, :]
                tmp3.append(1)
                count1 += 1
                t += 1
        elif target_vec[0, 70] == 1 or target_vec[0, 102] == 1:
            if count2 < N/2:
                zz = np.array(X[i])
                data[t] = zz[0, :]
                tmp3.append(0)
                count2 += 1
                t += 1
        if t >= N:
            break


def logit_1d(X, w):
    return 1 / (1 + np.exp(-X.dot(w)))


def logit_2d(X, w):
    tmp = np.einsum('ijk,ki->ij', X, w)
    return 1 / (1 + np.exp(-tmp))


def generate_data(N, dim, noise_ratio, save=False):
    # Generate data for logistic regression
    X = np.random.randn(N, dim)
    norm = np.sqrt(2 * np.linalg.norm(X.T.dot(X), 2) / N)
    X /= 2 * norm

    # Generate labels
    w_0 = np.random.rand(dim)
    Y = logit_1d(X, w_0)
    Y[Y > 0.5] = 1
    Y[Y <= 0.5] = 0

    noise = np.random.binomial(1, noise_ratio, N)
    Y_train = (noise - Y) * noise + Y * (1 - noise)

    if save:
        np.save('data/saved_data/' + str(N) + 'x.npy', X)
        np.save('data/saved_data/' + str(N) + 'y.npy', Y_train)

    return X, Y_train


def load_generated_data(N, basepath='data/saved_data/'):
    x_path = basepath + str(N) + 'x.npy'
    y_path = basepath + str(N) + 'y.npy'
    X = np.load(x_path)
    Y = np.load(y_path)
    return X, Y