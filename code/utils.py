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

def load_svm_data(K, with_label=[0, 8], reshape= False):
    
    train_set, valid_set, test_set = load_mnist()

    X_train, y_train = sample(3000, train_set[0], train_set[1], with_label=with_label)
    if len(with_label) == 2:
        y_train[np.where(y_train == with_label[0])] = -1
        y_train[np.where(y_train == with_label[1])] = 1
    X_dist, y_dist = distrute_dataset(K, X_train, y_train)

    X_test, y_test = sample(1000, test_set[0], test_set[1], with_label=with_label)
    if len(with_label) == 2:
        y_test[np.where(y_test == with_label[0])] = -1
        y_test[np.where(y_test == with_label[1])] = 1
    X_test_dist, y_test_dist = distrute_dataset(K, X_test, y_test)
    
    if reshape:
        for i in range(len(X_dist)):
            X_dist[i] = X_dist[i].reshape(X_dist[i].shape[0], 1, 28, 28)
        for i in range(len(X_test_dist)):
            X_test_dist[i] = X_test_dist[i].reshape(X_test_dist[i].shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    return X_dist, y_dist, X_test_dist, y_test_dist, X_test, y_test


def generate_AP(lam):
    side = 3 / np.sqrt(lam)  # The side of observation window
    Area = (2 * side) ** 2
    # Generate the BS number according to Poisson distribution
    Nb = np.random.poisson(lam * Area)
    
    return Nb

def db_to_linear(x):
    return 10 ** (x/10)