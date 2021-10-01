import numpy as np

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
