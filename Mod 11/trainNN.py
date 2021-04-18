from numpy import random, zeros, exp, clip, dot, log, sum, argmax, unique, arange, float
from sklearn.metrics import confusion_matrix
   
class NeuralNetMLP(object):
    def __init__(self, n_hidden=30, n_hidden2=20, epochs=100, eta=0.001, minibatch_size=1, seed=None):
        self.random = random.RandomState(seed)  # used to randomize weights
        self.n_hidden = n_hidden  # size of the hidden layer
        self.n_hidden2 = n_hidden2  # size of the hidden layer
        self.epochs = epochs  # number of iterations
        self.eta = eta  # learning rate
        self.minibatch_size = minibatch_size  # size of training batch - 1 would not work
    
    @staticmethod
    def onehot(y, n_classes):  # one hot encode the input class y
        onehot = zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.0
        return onehot.T
    
    @staticmethod
    def sigmoid(z):  # Eq 1
        return 1.0 / (1.0 + exp(-clip(z, -250, 250)))

    def _forward(self, X):  # Eq 2
        z_h = dot(X, self.w_h)
        a_h = self.sigmoid(z_h)
        z_h2 = dot(a_h, self.w_h2)
        a_h2 = self.sigmoid(z_h2)
        z_out = dot(a_h2, self.w_out)
        a_out = self.sigmoid(z_out)
        return z_h, a_h, z_h2, a_h2, z_out, a_out

    @staticmethod
    def compute_cost(y_enc, output):  # Eq 4
        term1 = -y_enc * (log(output))
        term2 = (1.0-y_enc) * log(1.0-output)
        cost = sum(term1 - term2)
        return cost

    def predict(self, X):
        z_h, a_h, z_h2, a_h2, z_out, a_out = self._forward(X)
        y_pred = argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        import sys
        n_output = unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden2, n_output))
        self.w_h2 = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, self.n_hidden2))
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))
        y_train_enc = self.onehot(y_train, n_output)  # one-hot encode original y
        for i in range(self.epochs):
            indices = arange(X_train.shape[0])
            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
                z_h, a_h, z_h2, a_h2, z_out, a_out = self._forward(X_train[batch_idx])
                sigmoid_derivative_h = a_h * (1.0-a_h)  # Eq 3
                sigmoid_derivative_h2 = a_h2 * (1.0-a_h2)  # Eq 3
                delta_out = a_out - y_train_enc[batch_idx]  # Eq 5
                delta_h2 = (dot(delta_out, self.w_out.T) * sigmoid_derivative_h2)  # Eq 6
                delta_h = (dot(delta_h2, self.w_h2.T) * sigmoid_derivative_h)  # Eq 6
                grad_w_out = dot(a_h2.T, delta_out)  # Eq 7
                grad_w_h2 = dot(a_h.T, delta_h2)  # Eq 8
                grad_w_h = dot(X_train[batch_idx].T, delta_h)  # Eq 8
                self.w_out -= self.eta*grad_w_out  # Eq 9
                self.w_h -= self.eta*grad_w_h  # Eq 9
                self.w_h2 -= self.eta*grad_w_h2  # Eq 9
            # Evaluation after each epoch during training
            z_h, a_h, z_h2, a_h2, z_out, a_out = self._forward(X_train)
            cost = self.compute_cost(y_enc=y_train_enc, output=a_out)
            y_train_pred = self.predict(X_train)  # monitoring training progress through reclassification
            y_valid_pred = self.predict(X_valid)  # monitoring training progress through validation
            train_acc = ((sum(y_train == y_train_pred)).astype(float) / X_train.shape[0])
            valid_acc = ((sum(y_valid == y_valid_pred)).astype(float) / X_valid.shape[0])
            sys.stderr.write('\r%d/%d | Cost: %.2f ' '| Train/Valid Acc.: %.2f%%/%.2f%% '%
                (i+1, self.epochs, cost, train_acc*100, valid_acc*100))
            sys.stderr.flush()
        #
        return self