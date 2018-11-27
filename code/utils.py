import gzip
import pickle
import numpy as np
import math
import torch
import torch.nn as nn

def to_np(x):
    return x.data.cpu().numpy()

def load_mnist():
    dataset = 'mnist.pkl.gz'
    f = gzip.open(dataset, 'rb')
    train, valid, test = pickle.load(f, encoding='bytes')
    f.close()
    return train, valid, test


def sample(n_samples, X, y, with_label=[]):
    '''
    with_label: select certains digist with give label. Used for 2-class SVM problem and linear regression problem.
    '''
    print('Sampling {} data samples from dataset.'.format(n_samples))
    if len(with_label) > 0:
        label_index = []
        for label in with_label:
            label_index += [i for i, x in enumerate(y) if x == label]
        X = X[label_index, :]
        y = y[label_index]

    index = np.random.choice(range(X.shape[0]), size=n_samples, replace=False)
    return X[index], y[index]



def distrute_dataset(K, X, y, form='uniform'):
    # distribute the dataset X, y to K nodes with difference ways
    X_dist = []
    y_dist = []
    N_samples = X.shape[0]
    if form is 'uniform':
        N_dist = math.floor(N_samples/K)
        print('Seperating total {} data samples to {} nodes with each containing {} samples'.format(N_samples, K, N_dist))
        for i in range(K):
            X_dist.append(X[i*N_dist:(i+1)*N_dist,:])
            y_dist.append(y[i*N_dist:(i+1)*N_dist])
    return X_dist, y_dist