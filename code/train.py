from models import *
from utils import *
import argparse
import matplotlib.pyplot as plt
import random

# set the training parameters
parser = argparse.ArgumentParser()
parser.add_argument("--c", type=float, default=0.01)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batchsize", type=int, default=10)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--prop_k", type=int, default=5)
parser.add_argument("--threshhold", type=float, default=5)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# set the random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)

# set the architecture parameters
K = 10  # number of local nodes
T = 200 # total number of training steps
tau = 10 # global aggregation frequency
in_dim = 784
out_dim = 1

def train(mode):

    # prepare dataset
    train_set, valid_set, test_set = load_mnist()
    with_label = [0, 8]  # which digit to select
    
    X_train, y_train = sample(1000, train_set[0], train_set[1], with_label=with_label)
    y_train[np.where(y_train == with_label[0])] = -1
    y_train[np.where(y_train == with_label[1])] = 1
    X_dist, y_dist = distrute_dataset(10, X_train, y_train)
    
    X_test, y_test = sample(1000, test_set[0], test_set[1], with_label=with_label)
    y_test[np.where(y_test == with_label[0])] = -1
    y_test[np.where(y_test == with_label[1])] = 1
    X_test_dist, y_test_dist = distrute_dataset(10, X_test, y_test)
    
    # initialize model
    fl_model = ScheduleFedLearn(K, in_dim, out_dim, args, method='SVM')

    # training process
    train_accuracy_list = []
    test_accuracy_list = []
    local_test_list = []
    for t in range(T):
        for k in range(K):
            fl_model.update_local(X_dist[k], y_dist[k], k)
        if (t % tau == 0):
            print('Perform global aggregation in step {}'.format(t))
            fl_model.global_aggregation(mode=mode, step=t)
            test_accuracy = fl_model.predict_global(X_test, y_test)
            print('Global test accuracy at step {} is {}'.format(t, test_accuracy))
            test_accuracy_list.append(test_accuracy)

        if t % 1 == 0:
            train_accuracy = []
            for k in range(K):
                train_accuracy.append(fl_model.predict_local(X_test_dist[k], y_test_dist[k], k))
            train_accuracy = np.sum(np.asarray(train_accuracy))/K
            #print('The accuracy in step {} is {}'.format(t, train_accuracy))
            train_accuracy_list.append(train_accuracy)

    
    # 
    # plt.figure()
    # plt.plot(test_accuracy_list)
    # plt.show()
    
    return test_accuracy_list, train_accuracy_list


test_accuracy_list1, train_accuracy_list1 = train('random')
test_accuracy_list2, train_accuracy_list2 = train('rrobin')
test_accuracy_list3, train_accuracy_list3 = train('prop_k')

# plt.figure()
# plt.plot(train_accuracy_list1)
# plt.plot(train_accuracy_list2)
# plt.plot(train_accuracy_list3)
# plt.legend(['random', 'rrboin', 'prop_k'])
# plt.xlabel('#Training steps')
# plt.ylabel('Train Accuracy')
# plt.show()

plt.figure()
plt.plot(test_accuracy_list1)
plt.plot(test_accuracy_list2)
plt.plot(test_accuracy_list3)
plt.legend(['random', 'rrboin', 'prop_k'])
plt.xlabel('#Training steps')
plt.ylabel('Test Accuracy')
plt.show()



