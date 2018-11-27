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
        X = torch.FloatTensor(X)
        predict = self.global_model(X).detach().numpy().squeeze()
        predict[np.argwhere(predict >= 0)] = 1
        predict[np.argwhere(predict < 0)] = -1
        accuracy = np.sum(predict == Y) / len(Y)

        return accuracy


class ScheduleFedLearn(object):
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

            # print("Node: {} Epoch: {:4d}  loss:{}".format(k, epoch, sum_loss / N))

    def global_aggregation(self, num_nodes=5, mode='random', threshhold=1, step=0):
        # do a global aggregation
        # nodes: the index of nodes to be updated, a list of integers
        m = Exponential(1)
        h = m.sample(torch.Size([self.K])).squeeze()

        #transfer = torch.where(h>=threshhold, torch.FloatTensor([1]), torch.FloatTensor([0]))

        # select which local update can be used for aggregation
        if mode == 'random':
            node = np.random.choice(range(self.K), size=num_nodes, replace=False)
        elif mode == 'rrobin':
            nodes = np.arange(self.K)
            batch_index = (step*self.args.batchsize)//self.K
            node = nodes[ batch_index * self.args.batchsize : (batch_index+1) * self.args.batchsize]
        elif mode == 'prop_k':
            node = np.argsort(h.numpy())[-self.args.prop_k:]
        else:
            node = np.arange(self.K)

        success_node = []
        for i in node:
            if h[i] >=threshhold:
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
        # prediction task based on the global params
        X = torch.FloatTensor(X)
        predict = self.local_model[k](X).detach().numpy().squeeze()
        predict[np.argwhere(predict >= 0)] = 1
        predict[np.argwhere(predict < 0)] = -1
        # print(predict)
        # print(Y)
        # print(predict==Y)
        # print(np.sum(predict==Y))
        accuracy = np.sum(predict == Y) / len(Y)

        return accuracy

    def predict_global(self, X, Y):
        X = torch.FloatTensor(X)
        predict = self.global_model(X).detach().numpy().squeeze()
        predict[np.argwhere(predict >= 0)] = 1
        predict[np.argwhere(predict < 0)] = -1
        accuracy = np.sum(predict == Y) / len(Y)

        return accuracy
        
