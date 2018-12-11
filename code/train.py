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
parser.add_argument("--prop_k", type=int, default=5)
parser.add_argument("--threshhold", type=float, default=20)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
args.threshhold = db_to_linear(args.threshhold)
print(args.threshhold)

# set the random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)

# set the architecture parameters
K = 30  # number of local nodes
T = 500 # total number of training steps
tau = 10 # global aggregation frequency
in_dim = 784
out_dim = 1

def train(mode, method, lam=1e-4):

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
    Nb = generate_AP(lam)
    fl_model = ScheduleFedLearn(K, args, Nb, lam=lam, alpha=4, in_dim=in_dim, out_dim=out_dim, method=method)

    # training process
    train_accuracy_list = []
    test_accuracy_list = []
    local_test_list = []
    success_nodes = []
    global_counter = 0
    for t in range(T):
        for k in range(K):
            fl_model.update_local(X_dist[k], y_dist[k], k)
        if (t % tau == 0):
            print('Perform global aggregation in step {}'.format(t))
            updated_nodes = fl_model.location_global_aggregation(mode=mode, step=global_counter)
            test_accuracy = fl_model.predict(X_test, y_test)
            print('Global test accuracy at step {} is {}'.format(t, test_accuracy))
            test_accuracy_list.append(test_accuracy)
            success_nodes += updated_nodes
            global_counter += 1

        # if t % 1 == 0:
        #     train_accuracy = []
        #     for k in range(K):
        #         train_accuracy.append(fl_model.predict(X_test_dist[k], y_test_dist[k], k))
        #     train_accuracy = np.sum(np.asarray(train_accuracy))/K
        #     #print('The accuracy in step {} is {}'.format(t, train_accuracy))
        #     train_accuracy_list.append(train_accuracy)
    #return test_accuracy_list, train_accuracy_list

    return success_nodes, test_accuracy_list


success_nodes1, test_accuracy_list1 = train('random', 'MNIST_CNN')
success_nodes2, test_accuracy_list2 = train('rrobin', 'MNIST_CNN')
success_nodes3, test_accuracy_list3 = train('prop_k', 'MNIST_CNN')

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

unique_elements1, counts_elements1 = np.unique(np.asarray(success_nodes1), return_counts=True)
unique_elements2, counts_elements2 = np.unique(np.asarray(success_nodes2), return_counts=True)
unique_elements3, counts_elements3 = np.unique(np.asarray(success_nodes3), return_counts=True)

print('random', unique_elements1, counts_elements1)
print('rrobin', unique_elements2, counts_elements2)
print('prop_k', unique_elements3, counts_elements3)


