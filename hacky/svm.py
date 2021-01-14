import numpy as np


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, n_iters=100, batch_size=10):
        n_train, dim = X.shape
        n_classes = int(np.max(y) + 1)
        print("N classes", n_classes)
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, n_classes)

        loss_history = []
        for i in range(n_iters):
            X_batch = None
            y_batch = None
            mask = np.random.choice(n_train, batch_size, replace=True)
            X_batch = X[mask]
            y_batch = y[mask]

            loss_, grad = loss(self.W, X_batch, y_batch, reg)
            loss_history.append(loss_)

            self.W = self.W - learning_rate * grad
            if i % 100 == 0:
                print("iteration %d / %d: loss %f" % (i, n_iters, loss_))
        return loss_history


def loss(W, X, y, reg):
    l = 0.0
    grad = np.zeros(W.shape)
    n_train = X.shape[0]
    delta = 1.0
    scores = W.dot(X)
    print("N train", n_train)
    print("y", y)
    correct_class_score = scores[np.arange(n_train), y.astype(int)]
    margins = np.maximum(
        0, scores - correct_class_score[:, np.newaxis] + delta)
    margins[np.arange(n_train), y] = 0
    l = np.sum(margins)
    l /= n_train
    l += 0.5 * reg * np.sum(W.T.dot(W))

    X_mask = np.zeros(margins.shape)
    X_mask[margins > 0] = 1

    count = np.sum(X_mask, axis=1)
    X_mask[np.arange(n_train), y] = -count

    grad = X.T.dot(X_mask)
    grad /= n_train
    grad += np.multiply(W, reg)

    return l, grad


TRAIN_PATH = "mnist/train.csv"
TEST_PATH = "mnist/mnist_test.csv"

train_o = np.loadtxt(TRAIN_PATH, skiprows=1, delimiter=",")
test_o = np.loadtxt(TEST_PATH, skiprows=1, delimiter=",")

train_labels = train_o[:, :1]
test_labels = test_o[:, :1]
print(train_labels.shape)
print(test_labels)

train_data = train_o[:, 1:]
test_data = test_o[:, 1:]

print(train_data.shape)
print(test_data.shape)

lc = LinearClassifier()
lh = lc.train(train_data, train_labels)
print(lh)