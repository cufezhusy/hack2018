from cnn_utils import *
import matplotlib.pyplot as plt
X_train_orig, Y_train_orig, X_test, Y_test_orig, classes , times = load_dataset_1d(datapath="../data_10000_flat_100_200/dataset.pickle")

N = 9
fig, axes = plt.subplots(nrows = 3 , ncols = 3)
i = 0
for ax in axes.flatten():
    ax.plot(times,X_test[i,:].reshape(X_test[i,:].shape[0]))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    i = i + 1
plt.show()

