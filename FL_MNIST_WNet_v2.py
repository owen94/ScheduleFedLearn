#!/usr/bin/env python
# coding: utf-8

# # Fed-Learning in Wireless Environment

# ## Import Libraries

# In[1]:

import pandas as pd
import numpy as np
from sklearn import svm, metrics
import math
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Loading 

# In[2]:


csv_path_train_data = 'csv/training_image.csv'
csv_path_train_label = 'csv/training_label.csv'
csv_path_test_data = 'csv/test_image.csv'
csv_path_test_label = 'csv/test_label.csv'

X_all_train = pd.read_csv(csv_path_train_data, header=None)
y_all_train = pd.read_csv(csv_path_train_label, header=None)
X_all_test = pd.read_csv(csv_path_test_data, header=None)
y_all_test = pd.read_csv(csv_path_test_label, header=None)


# ## Scaling Data 

# In[3]:


# Scale down the magnitude for better traning experience
X_all_train /= 255
X_all_test /= 255

X_all_train = X_all_train.as_matrix()
y_all_train = y_all_train.as_matrix()
X_all_test = X_all_test.as_matrix()
y_all_test = y_all_test.as_matrix()


# ## Hyperparameters

# In[145]:


n_train = 5000
n_test = 1000



T = 1000 # upper bound for iteration
tau = 10 # interval of global aggregation
eta = 0.01 # step size of gradient descent
lam = 0.01
K = 50 # total number of devices to be selected

modes = ["random", "sequential", "proport_k"] # how to select stable devices
number_to_select = 5 # the limit of the number of stable devices

# history_K = [5, 10, 15, 20, 25, 30] # number of devices
# log of training loss values, which element is compatible with mode. Each mode's element is compatible with the number of devises
all_history_loss = [[] for i in range(len(modes))] 
# log of training accuracies, which element is compatible with mode. Each mode's element is compatible with the number of devises
all_history_accuracy =  [[] for i in range(len(modes))] 


# ## System Parameters 

# In[156]:


lambda_b = 10**-4 # The access point density

SNR_u = 17 # in unit of dB
NB_0 = -90 # in unit of dB

Pmw = 10**-3

Put = Pmw * 10**(SNR_u/10) # power in the linear unit
sigma_2 = Pmw * 10**(NB_0/10) # power in the linear unit

alpha = 4 # path loss exponent

theta_dB = 10
theta = 10**(theta_dB/10)

Km = K


# ## Subsampling

# In[52]:


indexes_train_batch = np.random.choice(range(X_all_train.shape[0]), size=n_train, replace=False) # The np.random.choice gives a random selected vector from a range
indexes_test_batch = np.random.choice(range(X_all_test.shape[0]), size=n_test, replace=False)

X_train_batch = X_all_train[indexes_train_batch, :]
y_train_batch = y_all_train[indexes_train_batch]
X_test_batch = X_all_test[indexes_test_batch, :]
y_test_batch = y_all_test[indexes_test_batch]


# ## Distribute Data Set 

# In[53]:


Xs_train = [] # Xs_train is list containing feature data per device
ys_train = [] # ys_train is list containing labels per device

N_data = X_train_batch.shape[0]
N_X_train = math.floor(N_data/K) # number of data per device, assuming the number of data is the same for all  devices here

### shuffle X and y based on shuffled index
index_shuffle = np.arange(N_data)
np.random.shuffle(index_shuffle)
X_train_batch = X_train_batch[index_shuffle, :]
y_train_batch = y_train_batch[index_shuffle, :]

for k in range(K):
    X_train = X_train_batch[N_X_train*k:N_X_train*(k+1), :]
    y_train = y_train_batch[N_X_train*k:N_X_train*(k+1), :]
    Xs_train.append(X_train) # Note: Xs_train is a list with K elements, whereas each element is a matrix
    ys_train.append(y_train) # Similarly, ys_train is a list with K elements, whereas each element is a label


# ## Initialize 

# In[54]:


# initialize weight(global and local), Ws_local is list containing weight per device
# each row of W means all-vs-one classifier
# The set function eleminates the repeated elements, only the tags are remained
N_class = len(set(y_all_train.flatten())) 
N_features = X_train_batch.shape[1]

# For each class there is a specific feature, in this example, it is 8*784
W_global = np.random.rand(N_class, N_features)
Ws_local = [W_global for k in range(K)] # weights for all devices are the same at t = 0, 
                                        # Make a list of K and store the Ws_global piecewhilst


# ## Functions 

# In[55]:


# lam is penalty
def get_loss_local(W, X, y, lam):
    losses_class = [] ### list containing loss per classifier
    N_X = X.shape[0]
    N_class = len(set(y.flatten()))
    
    ### calculate loss per classifier
    for label in range(N_class):
        ### binarize y
        y_binary = np.copy(y) # initialize
        y_binary[y_binary==label] = 1
        y_binary[y_binary!=label] = -1
        
        w = W[label, :]
        
        loss = 0
        first_term = (lam/2) * np.linalg.norm(w, 2)
        for i in range(N_X):
            second_term = (1/2) * max(0, 1-y_binary[i]*np.dot(w, X[i, :].T))
            loss += second_term

        loss += first_term
        loss /= N_X
        losses_class.append(loss)
        
    total_loss = sum(losses_class) / N_class
    
    return total_loss


# In[19]:


# losses_local is list containing loss per device
def get_loss_global(losses_local, Xs_train):
    loss = 0
    N_data = 0
    K = len(Xs_train)# number of devices
    for k in range(K):
        loss += Xs_train[k].shape[0] * losses_local[k]
        N_data += Xs_train[k].shape[0]
        
    loss /= N_data
    
    return loss


# In[20]:


def get_path_loss_matrix(lambda_b, K, alpha_NL):
    
    alpha = alpha_NL
    Km = K
    
    side = 3/np.sqrt(lambda_b) # The side of observation window
    Area = (2*side)**2
    # Generate the BS number according to Poisson distribution
    Nb = np.random.poisson(lambda_b * Area)  # Number of BSs

    # The location of BSs, being complex numbers
    BS_loc = np.random.uniform(-side, side, (Nb, 1)) + 1j * np.random.uniform(-side, side, (Nb, 1))
    BS_loc = sorted(BS_loc, key=abs)  # sort according to locations, BS_loc[0] can be regarded as the typical

    # The UE locations, each UE locates inside the Voronoi cell of a BS
    UE_MatLoc = np.zeros((Nb, Km), dtype=np.complex_)
    UE_OcuIdx = [0 for i in range(Nb)]

    OcuSm = 0
    for iki in range(10**3) :

        if OcuSm == Nb*Km :
            # print('cool!')
            break

        tmp_ue = np.random.uniform(-side, side, 1) + 1j * np.random.uniform(-side, side, 1)
        dist = np.abs(BS_loc - tmp_ue)

        row_indx = np.argmin(dist)
        if UE_OcuIdx[row_indx] < Km:
            UE_MatLoc[row_indx, UE_OcuIdx[row_indx]] = tmp_ue
            UE_OcuIdx[row_indx] += 1
            OcuSm += 1

    D_Mat = np.abs( UE_MatLoc - BS_loc[0] )
    PL_Mat = D_Mat**(-alpha)
    
    return Nb, PL_Mat


# In[21]:


### get sum of gradient per class
### w is 1-D
def get_sum_gradient(w, X, y, label, lam):
    ### binarize y
    y_binary = np.copy(y) # initialize
    y_binary[y==label] = 1
    y_binary[y!=label] = -1
    
    N_X = X.shape[0]
    sum_gradient = 0
    
    first_term = lam * w
    for i in range(N_X):
        indicator = 1 - y_binary[i]*np.dot(w, X[i, :].T)
        if(indicator<=0):
            second_term = 0
        else:
            second_term = indicator*(-y_binary[i]*X[i, :]) # w is 1*n_features vector
        
        sum_gradient += first_term + second_term
        
    sum_gradient /= N_X
    
    return sum_gradient


# In[22]:


# each row of W means all-vs-one classifier
def update_weight(W, X, y, lam, eta):
    W_updated = np.copy(W) # initialize
    N_class = len(set(y.flatten()))
    for label in range(N_class):
        w = W[label, :]
        W_updated[label, :] = w - eta * get_sum_gradient(w, X, y, label, lam)
        
    return W_updated


# In[93]:


# Xs_train is list containing dataset per device, Ws_local is list containing the coordinates of the center per device
# this function is for unstable connection although global updates are distributed to all devices because server has an enough power to do that
def limited_global_aggregation(Xs_train_limited, Ws_local_limited, Ws_local, N_ap, h_vec, PL_Matrix):
    Nb = N_ap
    PL_Mat = PL_Matrix 
    
    K = len(Ws_local) # number of devices 
    K_limited = len(Xs_train_limited) # number of devices to get update
    N_data = 0 # initialize number of the whole data
    W_global = np.zeros(Ws_local_limited[0].shape) # initialize
    
    for k in range(K_limited):
        SucInd = 0
        # Generate the channel gain
        CG_vec = np.concatenate((h_vec[k].reshape(1,1), np.random.exponential(1,(Nb-1,1))))
        RecVec = np.multiply( PL_Mat[:,k].reshape(Nb,1), CG_vec )        
        
        SigPow = RecVec[0,0] # received signal strength
        IntFnc = np.sum(RecVec) - SigPow + sigma_2 # received interference
        
        SINR = SigPow/IntFnc # The SINR
        if SINR > theta: 
            SucInd = 1         
        
        N_X_train = Xs_train_limited[k].shape[0]
        W_global += N_X_train * Ws_local_limited[k] * SucInd
        N_data += N_X_train * SucInd

    if N_data > 0: 
        W_global /= N_data
        update_success = True
    else:
        update_success = False         
    
    Ws_local_updated = [W_global for k in range(K)] # update local weights based on global weight are distributed to all the devices
    return update_success, W_global, Ws_local_updated


# In[138]:


# select devices which can transmit updates to central server
# K is the number of all the devices
# mode can be "sequential" or "random"
# if mode is sequential, give count parameter to **option. count starts from 0, meaning the number of global aggregation
# return the numbers of selected devices(list)
# number_to_select means how many devices you will choose
def select_stable_devices(K, mode, channel, number_to_select, **option):
    CSI_vec = channel
    if mode=="random":
        selected_devices = np.random.choice(range(K), number_to_select, replace=False)
        selected_devices = selected_devices.tolist()
    elif mode=="sequential":
        begin_index = number_to_select*option['count'] % K
        end_index = number_to_select*(option['count']+1) % K
        if begin_index<end_index: # normal case
            selected_devices = list(range(K))[begin_index:end_index]
        else: # if the slicing exceed the length of list
            selected_devices = list(range(K))[begin_index:] +  list(range(K))[:end_index]
    elif mode=="proport_k":
        selected_devices = np.argsort(-CSI_vec[:,0])[0:number_to_select]
        selected_devices = selected_devices.tolist()        

    return selected_devices


# In[25]:


def predict(W, X):
    Y = np.dot(W, X.T) # Y : number of class * number of X
    y_predict = np.argmax(Y, axis=0)
    
    return y_predict


# ## Simulations

# In[150]:


count = 0 # how many times global updates are carried out
isim = 10**2 # number of simulation steps

all_history_loss_mat = np.zeros((len(modes),isim,T))
all_history_accuracy_mat = np.zeros((len(modes),isim,T))
y_predict_test = np.zeros((len(modes),1))


for i_isim in range(isim):

    print('It is the %r-th step out of %r' % (i_isim + 1, isim ) )

    # Generate the locations
    N_ap, PL_Matrix = get_path_loss_matrix(lambda_b, K, alpha)    
    
    for i, mode in enumerate(modes):  
                    
        for t in range(T):
            ### update weights at each device
            ### i.e., at every time stamp, each local devive will update its individual weight as: w_s,t --> w_s,{t+1}
            for k in range(K):
                Ws_local[k] = update_weight(Ws_local[k], Xs_train[k], ys_train[k], lam, eta)
            
            # Upon each update slot, select the devices and udpate the weight
            if(t%tau==0):
                # Generate the in-cell channel vector, in total K
                h_vec = np.random.exponential(1,(K,1))
                
                ### select devices from which updates will be sent to the central server
                if mode=="random":
                    devices_to_update = select_stable_devices(K, mode, h_vec, number_to_select)
                elif mode=="sequential":
                    devices_to_update = select_stable_devices(K, mode, h_vec, number_to_select, count=count)
                elif mode=="proport_k":
                    devices_to_update = select_stable_devices(K, mode, h_vec, number_to_select)
                        
                Xs_train_selected = [Xs_train[device] for device in devices_to_update]
                Ws_local_selected = [Ws_local[device] for device in devices_to_update]
                
                PL_Mat = (PL_Matrix.T[devices_to_update]).T
                
                CSI_vec = h_vec[devices_to_update]
                
                UpDt_Suc, W_global_rec, Ws_local_rec = limited_global_aggregation(Xs_train_selected, Ws_local_selected, Ws_local, N_ap, CSI_vec, PL_Matrix)
                if UpDt_Suc: # Only update when there is changes in the global
                    W_global = W_global_rec
                    Ws_local = Ws_local_rec                
                
                count += 1

            ### check loss and accuracy
            losses_local = []
            accuracies_local = []
            for k in range(K):
                ### loss
                loss_local = get_loss_local(Ws_local[k], Xs_train[k], ys_train[k], lam)
                losses_local.append(loss_local)

                ### accuracy
                y_predict = predict(Ws_local[k], Xs_train[k])
                accuracy = metrics.accuracy_score(ys_train[k], y_predict)
                accuracies_local.append(accuracy)

            ### loss
            loss_global = get_loss_global(losses_local, Xs_train)
            all_history_loss_mat[i, i_isim, t] = loss_global

            ### accuracy
            accuracy_global = sum(accuracies_local) / K
            all_history_accuracy_mat[i, i_isim, t] = accuracy_global

for i in range(len(modes)):            
            
    history_loss_global = all_history_loss_mat[i].sum(axis=0)/isim
    history_accuracy_global = all_history_accuracy_mat[i].sum(axis=0)/isim

    all_history_loss[i].append(history_loss_global)
    all_history_accuracy[i].append(history_accuracy_global)  

    y_pred_vec = predict(W_global, X_test_batch)
    y_predict_test[i] = metrics.accuracy_score(y_test_batch, y_pred_vec)


for y in y_predict_test:
    print(y)


colors = ["blue", "red", "green"]
labels = ["random", "rrobin", "proport_fair"]
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(all_history_loss)):
    ax.plot(all_history_loss[i][0], c=colors[i], label=labels[i])
    ax.legend()
plt.savefig('Figures/Loss_FL_MNIST.png')
plt.savefig('Figures/Loss_FL_MNIST.eps')

plt.xlabel('Iterations')
plt.ylabel('Loss function')

# ## Plot accuracy

# In[49]:

colors = ["blue", "red", "green"]
labels = ["random", "rrobin", "proport_fair"]
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(all_history_accuracy)):
    ax.plot(all_history_accuracy[i][0], c=colors[i], label=labels[i])
    ax.legend()
    plt.savefig('Figures/Accuracy_FL_MNIST.png')
    plt.savefig('Figures/Accuracy_FL_MNIST.eps')
    plt.xlabel('Iterations')
    plt.ylabel('Loss function')



