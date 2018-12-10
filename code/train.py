from models import *
from utils import *
import argparse
import matplotlib.pyplot as plt
import random

# set the training parameters
parser = argparse.ArgumentParser()
parser.add_argument("--c", type=float, default=0.01)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batchsize", type=int, default=10)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--prop_k", type=int, default=2)
parser.add_argument("--threshhold", type=float, default=0.3)
parser.add_argument("--seed", type=int, default=2)
args = parser.parse_args()

# set the random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)

# set the architecture parameters
K = 10  # number of local nodes
T = 1000 # total number of training steps
tau = 10 # global aggregation frequency
in_dim = 784
out_dim = 1

def train(mode, method):

    # prepare dataset
    if method is 'SVM':
        X_dist, y_dist, X_test_dist, y_test_dist = load_svm_data(K, with_label=[0, 8])
    elif method is 'MNIST_CNN':
        X_dist, y_dist, X_test_dist, y_test_dist, X_test, y_test = load_svm_data(K, with_label=[], reshape=True)
    elif method is 'CIFAR10_CNN':
        print('Need to be implemented. ')
    else: 
        print('Need to be implemented. ')
    
    # initialize model
    fl_model = ScheduleFedLearn(K, args, in_dim, out_dim, method=method)

    # training process
    train_accuracy_list = []
    test_accuracy_list = []
    local_test_list = []
    for t in range(T):
        for k in range(K):
            fl_model.update_local(X_dist[k], y_dist[k], k)
        if (t % tau == 0):
            print('Perform global aggregation in step {}'.format(t))
            fl_model.location_global_aggregation(mode=mode, step=t)
            test_accuracy = fl_model.predict(X_test, y_test)
            print('Global test accuracy at step {} is {}'.format(t, test_accuracy))
            test_accuracy_list.append(test_accuracy)

        if t % 1 == 0:
            train_accuracy = []
            for k in range(K):
                train_accuracy.append(fl_model.predict(X_test_dist[k], y_test_dist[k], k))
            train_accuracy = np.sum(np.asarray(train_accuracy))/K
            #print('The accuracy in step {} is {}'.format(t, train_accuracy))
            train_accuracy_list.append(train_accuracy)

    
    # 
    # plt.figure()
    # plt.plot(test_accuracy_list)
    # plt.show()
    
    return test_accuracy_list, train_accuracy_list


test_accuracy_list1, train_accuracy_list1 = train('random', 'MNIST_CNN')
test_accuracy_list2, train_accuracy_list2 = train('rrobin', 'MNIST_CNN')
test_accuracy_list3, train_accuracy_list3 = train('prop_k', 'MNIST_CNN')

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
plt.title('PPP threshhold = ' + str(args.threshhold) +
          ', prop_k = ' + str(args.prop_k) + ', #local nodes = ' + str(K))
plt.show()



