import random
import numpy as np
import csv

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.dtrain = X
        self.dlabels = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.dlabels.dtype)
        for i in range(num_test):
            distances = np.sum(np.abs(self.dtrain - X[i,:]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.dlabels[min_index]
            print(str(i) + " / " + str(num_test))
        return Ypred


TRAIN_PATH = "mnist/train.csv"
TEST_PATH = "mnist/mnist_test.csv"

train_o = np.loadtxt(TRAIN_PATH, skiprows=1, delimiter=",")
test_o = np.loadtxt(TEST_PATH, skiprows=1, delimiter=",")

train_labels = train_o[:,:1]
test_labels = test_o[:,:1]
print(train_labels.shape)
print(test_labels)

train_data = train_o[:,1:]
test_data = test_o[:,1:]

print(train_data.shape)
print(test_data.shape)

nn = NearestNeighbor()
nn.train(train_data, train_labels)
Yte_predict = nn.predict(test_data)

print("accuracy: %f" % (np.mean(Yte_predict == test_labels)))
