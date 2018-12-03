from utils import *
import torch.optim as optim
from torch.distributions.exponential import Exponential


class LinearSVM(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        h = self.fc(x)
        return h

class LogisticReg(nn.Module):
    def __init__(self):
        super(LogisticReg, self).__init__()


    def __forward(self, x):
        return x


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FedLearn(object):
    def __init__(self, K, in_dim, out_dim, args, method='SVM'):
        self.K = K
        self.method = method
        self.args = args
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init_architecture()
        self.init_optimizers()
        self.local_n_samples = np.zeros(self.K)

    def init_architecture(self):
        # initlize local nodes
        self.local_model = []
        for i in range(self.K):
            self.local_model.append(LinearSVM(self.in_dim, self.out_dim))
        self.global_model = LinearSVM(self.in_dim, self.out_dim)

    def init_optimizers(self):
        self.optims = []
        for i in range(self.K):
            self.optims.append(optim.SGD(self.local_model[i].parameters(), lr=self.args.lr))
        self.global_optim = optim.SGD(self.global_model.parameters(), lr=self.args.lr)

    def update_local(self, X, Y, k):
        # update a local model
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        N = len(Y)
        self.local_n_samples[k] = N

        model = self.local_model[k]
        optimizer = self.optims[k]
        model.train()

        for epoch in range(self.args.epoch):
            perm = torch.randperm(N)
            sum_loss = 0

            for i in range(0, N, self.args.batchsize):
                x = X[perm[i: i + self.args.batchsize]]
                y = Y[perm[i: i + self.args.batchsize]]

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                optimizer.zero_grad()
                output = model(x)
                loss = torch.mean(torch.clamp(1 - output.t() * y, min=0))
                loss += self.args.c * torch.mean(model.fc.weight ** 2)  # l2 penalty
                loss.backward()
                optimizer.step()

                sum_loss += to_np(loss)

            #print("Node: {} Epoch: {:4d}  loss:{}".format(k, epoch, sum_loss / N))

    def global_aggregation(self, node=None):
        # do a global aggregation
        # nodes: the index of nodes to be updated, a list of integers
        params = []
        if node is None:
            node = np.arange(self.K)
        N_samples = np.sum(self.local_n_samples[node])
        if N_samples ==0:
            print('Please allocate samples to node before global aggregation. ')

        for i in node:
            param = self.local_model[i].state_dict()
            param.update((x, y * self.local_n_samples[i]) for x, y in param.items())
            params.append(list(param.values()))

        global_params = [sum(x) / N_samples for x in zip(*params)]

        # set the parameters for the global model
        for i, f in enumerate(self.global_model.parameters()):
            f.data = global_params[i]

        # set the parameters for the local model
        for k in node:
            for i, f in enumerate(self.local_model[k].parameters()):
                f.data = global_params[i]

    def predict_local(self, X, Y, k=None):
        # prediction task based on the global params
        self.local_model[k].eval()
        X = torch.FloatTensor(X)
        predict = self.local_model[k](X).detach().numpy().squeeze()
        predict[np.argwhere(predict>=0)] = 1
        predict[np.argwhere(predict<0)] = -1
        # print(predict)
        # print(Y)
        # print(predict==Y)
        # print(np.sum(predict==Y))
        accuracy = np.sum(predict==Y)/len(Y)

        return accuracy
    
    def predict_global(self, X, Y):
        self.global_model.eval()
        X = torch.FloatTensor(X)
        predict = self.global_model(X).detach().numpy().squeeze()
        predict[np.argwhere(predict >= 0)] = 1
        predict[np.argwhere(predict < 0)] = -1
        accuracy = np.sum(predict == Y) / len(Y)

        return accuracy


class ScheduleFedLearn(object):
    def __init__(self, K,  args, in_dim=None, out_dim=None, method='SVM'):
        '''
        :param K: number of local nodes
        :param in_dim: input dim for SVM or logistic regression
        :param out_dim: output dime for SVM or logistic regression
        :param args: parser arguments
        :param method: SVM, MNIST_CNN, CIFAR10_CNN, LogRegression
        '''
        self.K = K
        self.method = method
        self.args = args
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init_architecture()
        self.init_optimizers()
        self.to_device()
        self.local_n_samples = np.zeros(self.K)

    def init_architecture(self):
        # initlize local nodes
        self.local_model = []
        if self.method is 'SVM':
            for i in range(self.K):
                self.local_model.append(LinearSVM(self.in_dim, self.out_dim))
            self.global_model = LinearSVM(self.in_dim, self.out_dim)
        elif self.method is 'MNIST_CNN':
            for i in range(self.K):
                self.local_model.append(MNIST_CNN())
            self.global_model = MNIST_CNN()
        elif self.method is 'CIFAR10_CNN':
            for i in range(self.K):
                self.local_model.append(CIFAR10_CNN())
            self.global_model = CIFAR10_CNN()
        elif self.method is 'LogRegression':
            for i in range(self.K):
                self.local_model.append(LogisticReg())
            self.global_model = LogisticReg()
        else:
            print('Method not implemented........')

    def init_optimizers(self):
        self.optims = []
        for i in range(self.K):
            self.optims.append(optim.SGD(self.local_model[i].parameters(), lr=self.args.lr))
        self.global_optim = optim.SGD(self.global_model.parameters(), lr=self.args.lr)

    def to_device(self):
        self.device = torch.device("cuda" if use_cuda else "cpu")
        for i in range(self.K):
            self.local_model[i].to(self.device)
        self.global_model.to(self.device)

    def update_local(self, X, Y, k):
        # update a local model
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        N = len(Y)
        self.local_n_samples[k] = N

        model = self.local_model[k]
        optimizer = self.optims[k]
        model.train()

        for epoch in range(self.args.epoch):
            perm = torch.randperm(N)
            sum_loss = 0

            for i in range(0, N, self.args.batchsize):
                x = X[perm[i: i + self.args.batchsize]]
                y = Y[perm[i: i + self.args.batchsize]]

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                optimizer.zero_grad()
                output = model(x)
                if self.method is 'SVM':
                    loss = torch.mean(torch.clamp(1 - output.t() * y, min=0))
                    loss += self.args.c * torch.mean(model.fc.weight ** 2)  # l2 penalty
                elif self.method is 'MNIST_CNN':
                    loss = F.nll_loss(output, y)
                elif self.method is 'CIFAR10_CNN':
                    loss = F.nll_loss(output, y)
                elif self.method is 'LogisticReg':
                    loss =  F.smooth_l1_loss(output, y)

                else:
                    print('Method not implemented.... ')
                    
                loss.backward()
                optimizer.step()

                sum_loss += to_np(loss)

            # print("Node: {} Epoch: {:4d}  loss:{}".format(k, epoch, sum_loss / N))

    def global_aggregation(self, mode='random', step=0):
        # do a global aggregation
        # nodes: the index of nodes to be updated, a list of integers
        m = Exponential(1)
        h = m.sample(torch.Size([self.K])).squeeze()

        #transfer = torch.where(h>=threshhold, torch.FloatTensor([1]), torch.FloatTensor([0]))

        # select which local update can be used for aggregation
        if mode == 'random':
            node = np.random.choice(range(self.K), size=self.args.prop_k, replace=False)
        elif mode == 'rrobin':
            nodes = np.arange(self.K)
            batch_index = (step*self.args.prop_k)%self.K
            node = nodes[ batch_index * self.args.prop_k : (batch_index+1) * self.args.prop_k]
        elif mode == 'prop_k':
            node = np.argsort(h.numpy())[-self.args.prop_k:]
        else:
            node = np.arange(self.K)

        success_node = []
        for i in node:
            if h[i] >= self.args.threshhold:
                success_node.append(i)

        # only aggregatiate the nodes taht has a good channle state
        params = []
        N_samples = np.sum(self.local_n_samples[success_node])
        if N_samples == 0:
            print('ALL NODES FAILED!!!')
        else:
            for i in success_node:
                param = self.local_model[i].state_dict()
                param.update((x, y * self.local_n_samples[i]) for x, y in param.items())
                params.append(list(param.values()))

            global_params = [sum(x) / N_samples for x in zip(*params)]

            # set the parameters for the global model
            for i, f in enumerate(self.global_model.parameters()):
                f.data = global_params[i]

            # set the parameters for the local model
            for k in node:
                for i, f in enumerate(self.local_model[k].parameters()):
                    f.data = global_params[i]

    def predict_local(self, X, Y, k=None):
        X = torch.FloatTensor(X).to(self.device)
        if k is not None:
            self.local_model[k].eval()
            # prediction task based on the global params
            predict = self.local_model[k](X).detach().numpy().squeeze()
        else:
            self.global_model.eval()
            predict = self.global_model(X).detach().numpy().squeeze()
        if self.method is 'SVM':
            predict[np.argwhere(predict >= 0)] = 1
            predict[np.argwhere(predict < 0)] = -1
            accuracy = np.sum(predict == Y) / len(Y)
        elif self.method is 'MNIST_CNN' or self.method is 'CIFAR10_CNN':
            # Future version should use a test loader here instead of one batch
            Y.to(self.device)
            test_loss += F.nll_loss(prdict, reduction='sum').item()  # sum up batch loss
            pred = predict.max(1, keepdim=True)[1]  # get the index of the max log-probability
            accuracy = pred.eq(Y.view_as(pred)).sum().item()
            accuracy /= len(Y)
        elif self.method is 'LogisticReg':
            print('To be implemented')
        else:
            print('Test method not implemented')

        return accuracy

    def predict_global(self, X, Y):
        self.global_model.eval()
        X = torch.FloatTensor(X)
        predict = self.global_model(X).detach().numpy().squeeze()
        predict[np.argwhere(predict >= 0)] = 1
        predict[np.argwhere(predict < 0)] = -1
        accuracy = np.sum(predict == Y) / len(Y)

        return accuracy
        
