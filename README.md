# ScheduleFedLearn
Federated Learning with Schedule Policy 

### Prerequist 
Please install pytorch, numpy and matplotlib via pip. 
In macos without cuda, run: ```pip3 install torch torchvision```

### Test 
run ```python/python3 train.py``` for initial results. 

There are 3 different scheduling policies to be evaluated: 

- Random scheduling 'random': Uniformly sample 10 out of 50 at random each time, and update the selected 10;
- Round robin 'rrobin': label the devices from 1 to 50, first round update 1~10, second round 11~20, third round 21~30, ..., rotate around these groups;
- Strongest channel (approximatedly proportional fair) 'prop_k': From the (h_1, h_2, ..., h_50), select the 10 devices with the largest h_i and update their parameters.

### To do list

1. Implement other ML algorithms: logistic regression, k-means, and CNN
2. Compare the three scheduling policies and plot the results. 
3. Find more datasets for experiments
4. Revise the major version (mostly done) 

