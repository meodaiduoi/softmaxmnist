import matplotlib
import matplotlib.pyplot as plt
import numpy as np # linear algebra
from mnist import MNIST

from time import perf_counter
import os


#
os.path.abspath('./mnistdb')

mnistdt = MNIST(os.path.abspath('./mnistdb'))
img_arr_test, lb_arr_test = mnistdt.load_training()
img_arr_train, lb_arr_train = mnistdt.load_testing()

# image view
def resize(img):
    return np.reshape(img, [-1, 28, 28, 1])

def image_show(a):
    some_digit = a
    some_digit_image = some_digit.reshape(28,28)
    plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")
    plt.axis("off")

# reusualbe function
def normalize(X):
    return X/255

def onehot_encoding(y):
    return np.eye(np.max(y)+1)[y]

def add_bias(X):
    return np.insert(X, 0, 1, axis=1)

def init_weight(X_col_size, y_size):
    return np.zeros(shape=(y_size, X_col_size -1)) # X_col_size not include bias row

#
class MultiClassLogisticRegression:
    def __init__(self, iteration, learning_rate, accuracy):
        self.n_iter = iteration     #Vong lap
        self.isBreak =  0.0001     #do chenh lenh dung
        self.lr =  learning_rate  #Learning rate
        self.i_Acc = accuracy

    def fit(self, X_train, y_train, X_test, y_test):
        self.classes = np.unique(y)   #lay ra all cac du lieu khac nhau cua y giong nhu set
        self.class_labels = {c:i for i,c in enumerate(self.classes)}   #dan nhan cho tuong class
        X, y = self.ADCT(X, y)  #CT trong giao trinh
        X_val, y_val = self.ADCT(X_test, y_test)
        self.loss_train = []
        self.error_train = []
        self.loss_test = []
        self.error_test = []
        self.weights = np.zeros(shape=(len(self.classes),X.shape[1]))  #(10x784)
        self.gradient_descent(X_train, y_train,  X_test, y_test)
        return self

    def ADCT(self, x, y):
        x = self.add_bias(x)  #CT trong giao trinh
        y = self.one_hot(y)   #CT trong giao trinh
        return x, y

    def gradient_descent(self, X_train, y_train, X_test, y_test):
          start = perf_counter()
          j = 0
          for i in range(self.n_iter):

            self.multinomial_logreg_error(X_train, y_train, X_test, y_test)
            update = self.multinomial_logreg_grad_i(X_train, y_train)
            self.weights += update
            if np.abs(update).max() < self.isBreak:
                break
            if i % self.i_Acc == 0:

                message = 'Loop {},  Loss_train {},  Acc_train {}, Loss_test {}, Acc_test {}'.format(i+1,self.loss_train[i], self.error_train[i], self.loss_test[i], self.error_test[i])
                j += 1
                print(message)
          end = perf_counter()
          time = int(end) - int(start)
          print('Chuong trinh chay mat: {} s'.format(time))

    def multinomial_logreg_error(self, X_train, y_train, X_test, y_test):
          self.loss_train.append(self.multinomial_logreg_loss_i(y_train, self.predict(X_train)))
          self.error_train.append(self.evaluate(X_train, y_train))
          self.loss_test.append(self.multinomial_logreg_loss_i(y_test, self.predict(X_test)))
          self.error_test.append(self.evaluate(X_test, y_test))

    def predict(self, X):
        pre_vals = np.dot(X, self.weights.T).reshape(-1,len(self.classes))
        return self.softmax(pre_vals)

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)

    def CheckModel(self, X, position):
        print('Dự đoán cua X_test[{}] la: {}'.format(position, self.predict_class(X)[position]))

        plt.imshow(X[position].reshape(28,28), cmap='gray')
        plt.show()

    def predict_class(self, X):
        X_add = self.predict(self.add_bias(X))   #Xac suat
        return np.argmax(X_add, axis=1)

    def add_bias(self,X):
        return np.insert(X, 0, 1, axis=1)

    def one_hot(self, y):
        return np.eye(len(self.classes))[(y).reshape(-1)]

    def evaluate(self, X, y):
        return np.mean(np.argmax(self.predict(X), axis=1) == np.argmax(y, axis=1))

    #cross_entropy
    def multinomial_logreg_loss_i(self, y, probs):
        return -(np.mean(y * np.log(probs)))

    def score(self, X, y):
        return np.mean(self.predict_class(X) == y)

    def multinomial_logreg_grad_i(self, X, y):
        error = y - self.predict(X)
        return (self.lr * np.dot(error.T, X))