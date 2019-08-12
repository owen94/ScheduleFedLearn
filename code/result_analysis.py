import numpy as np
import matplotlib.pyplot as plt
import os
db = 20
mode = 'SVM'

path = '../results/'
if not os.path.exists(path):
    os.mkdir(path)

test_rr = np.load(os.path.join(path, "test_rr_" + str(db) + "_" + mode + ".npy"))
test_rs = np.load(os.path.join(path, "test_rs_" + str(db) + "_" + mode + ".npy"))
test_pf = np.load(os.path.join(path, "test_pf_" + str(db) + "_" + mode + ".npy"))


plt.figure()

for i in range(0, len(test_rr)):
    plt.plot(test_pf[i])
    print(test_pf[i] - test_rr[i])

plt.show()