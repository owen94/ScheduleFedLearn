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
parser.add_argument("--threshhold", type=float, default=30)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
db = args.threshhold
args.threshhold = db_to_linear(args.threshhold)
print(args.threshhold)

# set the random seeds
# move to the main function to test multiple seeds

# set the architecture parameters
K = 100  # number of local nodes
T = 500 # total number of training steps
tau = 10 # global aggregation frequency
in_dim = 784
out_dim = 1

def train(mode, method, lam=1e-4):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(1234)
    # prepare dataset
    if method is 'SVM':
        X_dist, y_dist, X_test_dist, y_test_dist, X_test, y_test = load_svm_data(K, with_label=[0, 8],n_samples=K*5 )
    elif method is 'MNIST_CNN':
        X_dist, y_dist, X_test_dist, y_test_dist, X_test, y_test = load_svm_data(K, with_label=[], reshape=True, n_samples=K*30)
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


if __name__ == '__main__':



    seed_list = np.arange(10)
    test_rr = []
    test_rs = []
    test_pf = []
    mode = 'MNIST_CNN'

    for seed in seed_list:
        args.seed = seed
        success_nodes1, test_accuracy_list1 = train('random', mode)
        success_nodes2, test_accuracy_list2 = train('rrobin', mode)
        success_nodes3, test_accuracy_list3 = train('prop_k', mode)

        test_rs.append(test_accuracy_list1)
        test_rr.append(test_accuracy_list2)
        test_pf.append(test_accuracy_list3)

    plt.figure()
    mean_test_rs = np.mean(np.asarray(test_rs), axis=0)
    mean_test_rr = np.mean(np.asarray(test_rr), axis=0)
    mean_test_pf = np.mean(np.asarray(test_pf), axis=0)
    plt.plot(mean_test_rs)
    plt.plot(mean_test_rr)
    plt.plot(mean_test_pf)
    plt.legend(['random', 'rrboin', 'prop_k'])
    plt.xlabel('#Training steps')
    plt.ylabel('Test Accuracy')
    plt.title('PPP threshhold = ' + str(db) +
              'dB, prop_k = ' + str(args.prop_k) + ', #local nodes = ' + str(K))
    plt.show()

    unique_elements1, counts_elements1 = np.unique(np.asarray(success_nodes1), return_counts=True)
    unique_elements2, counts_elements2 = np.unique(np.asarray(success_nodes2), return_counts=True)
    unique_elements3, counts_elements3 = np.unique(np.asarray(success_nodes3), return_counts=True)

    print('random', unique_elements1, counts_elements1)
    print('rrobin', unique_elements2, counts_elements2)
    print('prop_k', unique_elements3, counts_elements3)

    # save the results

    path = '../results/'
    if not os.path.exists(path):
        os.mkdir(path)

    np.save(os.path.join(path, "test_rr_" + str(db) + "_" + mode + ".npy"), test_rr)
    np.save(os.path.join(path, "test_rs_" + str(db) + "_" + mode + ".npy"), test_rs)
    np.save(os.path.join(path, "test_pf_" + str(db) + "_" + mode + ".npy"), test_pf)



