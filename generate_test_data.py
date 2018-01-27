# ===================================================================================
# The code to generate test data used for Hack2018
# ===================================================================================


import matplotlib.pyplot as plt
import datetime
import numpy as np
import random
import pickle
from scipy import ndimage
from tqdm import tqdm


# outlier ts
def outlier_ts(x):
    y = np.random.randn(x.shape[0])
    pos = random.randint(1, x.shape[0]-1)
    y[pos] = y[pos] + random.randint(10,20)*np.random.choice([-1,1]) # apply a random shock
    return y

def jump_ts(x):
    y = np.random.randn(x.shape[0])
    pos = random.randint(1, x.shape[0]-1)
    y[pos:x.shape[0]] = y[pos:x.shape[0]] + random.randint(10,20) # apply a constant jump
    return y

def normal_ts(x):
    y = np.random.randn(x.shape[0])
    return y

def flat_start(x):
    rand_to = len(x) - np.random.randint(5,len(x)-20,dtype=int)
    return rand_to

def flat_stop(m,k):
    rand_stop = np.random.randint(m,k,dtype=int)
    return rand_stop

def get_flat_ts(x):
    y = normal_ts(x)
    start =  np.random.randint(5,len(x)-100)
    stop = start + np.random.randint(3,10)
    for j in range(start,stop):
        y[j] = y[j-1]
    return y

if __name__ == '__main__':
    N = 10000 # total size of the sample
    outlier_prob = .2 # Prob of data have outlier
    jump_prob = .2 # Prob of data that have jump ( not outlier)
    flat_prob = 0.2 # prob of data having flats

    # ===================================================================================
    # Remove the figures on the data folder
    # ===================================================================================
    import os
    mydir = 'data_10000_flat/'
    os.makedirs(mydir, exist_ok=True)
    def remove_all_png_file(mydir):
        filelist = [f for f in os.listdir(mydir) if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join(mydir, f))
    remove_all_png_file(mydir)


    X = []
    Y = []
    X_real_data = []
    # ===================================================================================
    # 1 Year data
    # ===================================================================================
    datelist = np.array([datetime.datetime(2017, 3, 1) + datetime.timedelta(days=i) for i in range(1, 300)])

    print("Now we start generate timeseires and add outlier randomly")
    for i in tqdm(range(0,N)):
        temp = random.random() # generate a random number
        if temp < outlier_prob:
            ts_type = 1
            timeseries = outlier_ts(datelist)
            name= str(i) + '_outlier'
        elif temp < outlier_prob + jump_prob:
            ts_type = 2
            timeseries = jump_ts(datelist)
            name = str(i) + '_jump'
        elif temp < outlier_prob + jump_prob + flat_prob:
            ts_type = 3
            timeseries = get_flat_ts(datelist)
            name = str(i) + '_flat'
        else:
            ts_type = 0
            timeseries = normal_ts(datelist)
            name = str(i) + '_normal'

        X_real_data.append(timeseries)

        # output figure
        #plt.figure(figsize=(6,1),dpi = 80)
        #plt.plot(datelist, timeseries)
        #fname = mydir + '%s.png'%name
        #plt.savefig(fname)
        #plt.close()

        # add the binary data to list
        #image = np.array(ndimage.imread(fname, flatten=False))
        #X.append(image)
        Y.append(ts_type)


    # ===================================================================================
    # divide the training the testing data
    # We say 90% should be in training set and 10% in testing set
    # ===================================================================================
    print("Cut the training and test set and then pickle")
    cut = int(N*0.95)
    dataset = {}
    dataset["train_set_x"] = X[0:cut]
    dataset["train_set_y"] = Y[0:cut]
    dataset["real_data_train"] = X_real_data[0:cut]
    dataset["test_set_x"] = X[cut:]
    dataset["real_data_test"] = X_real_data[cut:]
    dataset["test_set_y"] = Y[cut:]
    dataset["list_classes"] = [0,1,2,3]
    dataset["X_real_data"] = X_real_data
    dataset["time"] = datelist
    with open(mydir+'dataset.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(dataset, f)

    #remove_all_png_file(mydir)


