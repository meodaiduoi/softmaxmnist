import matplotlib
import matplotlib.pyplot as plt
import numpy as np # linear algebra

import os # path processing
from mnist import MNIST # mnist file reader
from time import perf_counter # timer

# Path Debuging
os.chdir(os.path.dirname(__file__ ))
print(os.getcwd())
print(os.path.abspath(""))

file_loc = os.path.abspath('')
mnistdt = MNIST(file_loc +'./mnistdb')
img_arr_test, lb_arr_test = mnistdt.load_testing()
img_arr_train, lb_arr_train = mnistdt.load_training()

# data fomater
def normalize(X):
    return X/255

#original
X_train = normalize(np.array(img_arr_train))
X_test = normalize(np.array(img_arr_test))
y_train = np.array(lb_arr_train)
y_test = np.array(lb_arr_test)
print("Du lieu x sau khi cau truc lai", X_train.shape)

# Hello Thầy
# Log
from datetime import datetime
class log:
    def __init__(self, name = "temp"):
        self.time = datetime.now()
        self.file_timeformat = self.time.strftime("_%d-%m-%Y_%H-%M")
        self.filename = os.path.join(os.path.abspath("./logs"), name + self.file_timeformat + '.txt')
        self.file = open(self.filename, "w+")

    def write(self, text):
        self.file.write(text)

    def writel(self, text):
        self.file.write("\n" + text)

    def close(self):
        self.file.close()

# save the model to disk
import pickle # save train model
def save_model(obj, filename):
    filename = os.path.join(os.path.abspath('./pretrain_models/'), filename + ".sav")
    pickle_file = open(filename, 'wb')
    pickle.dump(obj, pickle_file)
    pickle_file.close()

def load_model(filename):
    file_path = os.path.join(os.path.abspath('./pretrain_models/'), filename)
    loaded_model = pickle.load(open(file_path, 'rb'))
    return loaded_model

# Visualize data
def Visualize(loss_train, acc_train, loss_test, acc_test, filename):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(loss_train, 'tab:red')
    axs[0, 0].set_title('Loss train')
    axs[0, 1].plot(acc_train, 'tab:green')
    axs[0, 1].set_title('Acc train')
    axs[1, 0].plot(loss_test, 'tab:red')
    axs[1, 0].set_title('Loss test')
    axs[1, 1].plot(acc_test, 'tab:green')
    axs[1, 1].set_title('Acc test')
    fig.tight_layout()
    file_path = plt.savefig(os.path.join(os.path.abspath("./plots/"), filename))
    print('save plot thanh cong')


class Model:

    def __init__(self, n_iter=100, learning_rate = 1.0, step_print = 10, l2_param=0.0001,  n_slice = 1000, is_slice = True, filename = 'test.txt'):
        """
          param:
              n_iter: so vong lap
              learning_rate: 0.0001
              step_print: Khoang cach so lan in
              l2_param: fix overfit
              alpha: 1.0
              n_slice: So du lieu duoc cat
              is_clice: return True, False (co cat data hay khong?)
        """
        self.n_iter = n_iter
        self.l2_param = l2_param
        self.step_print = step_print
        self.learning_rate = learning_rate
        self.n_slice = n_slice
        self.is_slice = is_slice
        self.filename = filename
    
    def fit(self, X_train, y_train, X_test, y_test):
        np.random.seed(1024)
        self.classes = np.unique(y_train)
        self.n_classes = len(self.classes)
        X_train, y_train = self.ADCT(X_train, y_train)
        X_test, y_test = self.ADCT(X_test, y_test)
        self.loss_train= []
        self.acc_train = []
        self.loss_test = [] 
        self.acc_test = []
        self.weights = np.zeros(shape=(self.n_classes, X_train.shape[1]))  #(10x784) 
        self.gradient_descent(X_train, y_train, X_test, y_test)
        return self

    def ADCT(self, x, y):
        x = self.add_bias(x)  #CT trong giao trinh
        y = self.one_hot(y)   #CT trong giao trinh
        return x, y

    #*******************Part2********************#
    def gradient_descent(self, X_train, y_train, X_test, y_test):
        #print logs
        loger = log(self.filename)
        start = perf_counter()
        total_grad = []
        for i in range(self.n_iter):
            self.multinomial_logreg_error(X_train, y_train, X_test, y_test)

            if self.is_slice:
              #slice data
              idx = np.random.choice(X_train.shape[0], self.n_slice)
              X_slice, y_slice = X_train[idx], y_train[idx]
               #*******************Part1********************#
              grad = self.multinomial_logreg_grad_i(X_slice, y_slice)
              total_grad.append(grad)
            else:
              #full data
               #*******************Part1********************#
              grad = self.multinomial_logreg_grad_i(X_train, y_train)
              total_grad.append(grad)

            #update
            #overfit
            self.weights -= 1/X_train.shape[0] * self.learning_rate * (grad + (self.l2_param/2 * np.linalg.norm(self.weights)) )
            # self.weights -= self.learning_rate*grad
            if np.abs(grad).max() < self.l2_param: 
                #stop
                break
            if i % self.step_print == 0: 
                #print message
                message = 'Loop {}/{},  Loss_train {},  Acc_train {}, Loss_test {}, Acc_test {}'.format(i+1,self.n_iter + 1 ,self.loss_train[i-1], self.acc_train[i-1], self.loss_test[i-1], self.acc_test[i-1])
                loger.writel(message)
                print(message)
        end = perf_counter()
        time = int(end) - int(start)
        loger.close()
        print('Save log')
        self.multinomial_logreg_total_grad(total_grad)
        self.multinomial_logreg_total_grad(self.loss_train)
        print('Chuong trinh chay mat: {} s'.format(time))

    #*******************Part4********************#
    def multinomial_logreg_error(self, X_train, y_train, X_test, y_test):
          self.loss_train.append(self.multinomial_logreg_loss_i(y_train, self.pro_predict(X_train)))
          self.acc_train.append(self.multinomial_logreg_acc_i(X_train, y_train))   #ghi vao file .txt
          self.loss_test.append(self.multinomial_logreg_loss_i(y_test, self.pro_predict(X_test)))
          self.acc_test.append(self.multinomial_logreg_acc_i(X_train, y_train))

    #*******************Part1********************#
    def multinomial_logreg_loss_i(self, y, probs):
        #cross_entropy
        return -np.mean(y * np.log(probs))
    
    #*******************Part1********************#
    def multinomial_logreg_grad_i(self, X, y):
        # cost function
        error = self.pro_predict(X) - y
        grad = np.dot(error.T, X)
        return grad

    #*******************Part2********************#
    # def multinomial_logreg_total_grad (self, arr_grad):
    #     for i in range(len(arr_grad)):
    #       sum_grad += arr_grad[i]
    #     return sum_grad

    #*******************Part3********************#
    def multinomial_logreg_total_loss(self, arr_loss):
        return np.sum(arr_loss)

     #*******************Part3********************#
    def multinomial_logreg_total_grad(self, arr_grad):
        return np.sum(arr_grad)
        
    def multinomial_logreg_acc_i(self, X, y):
        # evaluate check acc
        return np.mean(np.argmax(self.pro_predict(X), axis=1) == np.argmax(y, axis=1))

    def pro_predict(self, X):
        score = np.dot(X, self.weights.T).reshape(-1,len(self.classes))
        result = self.softmax(score)
        return result
    
    def softmax(self, score):
        # result: [0->1]
        return np.exp(score) / np.sum(np.exp(score), axis=1).reshape(-1,1)
  
    def add_bias(self,X):
        return np.insert(X, 0, 1, axis=1)

    def one_hot(self, y):
        return np.eye(len(self.classes))[(y).reshape(-1)]

    def predict_layer(self, X):
        X_add = self.pro_predict(self.add_bias(X))   #Xac suat
        return np.argmax(X_add, axis=1)

    def sroce_pridict(self, X, y):
        return (np.mean(self.predict_layer(X) == y))

    def CheckModel(self, X, n_test):
        #Check and visualize
        fig = plt.figure(figsize=(20,10))
        print('train thu {} data'.format(n_test))
        for i in range(n_test):
            ax = fig.add_subplot(3, 5, i+1)
            pos = np.random.randint(X.shape[0])
            ax.title.set_text('Dự đoán X_test[{}] la: {}'.format(pos, self.predict_layer(X)[pos]))
            ax.imshow(X_test[pos].reshape(28,28), cmap='gray')
            plt.axis()

train_full = Model(is_slice=False, filename='train_full')
train_full.fit(X_train, y_train, X_test, y_test)
#Save model
save_model(train_full, "train_full")

#Load model
train_full = load_model("train_full.sav")
train_full.CheckModel(X_test, 10)
print('du doan dung ', train_full.sroce_pridict(X_train, y_train)*100, '%')

# slice 100 data train
train_100 = Model(n_iter=100 , is_slice=True, step_print=10)
train_100.fit(X_train, y_train, X_test, y_test)
#Save model
save_model(train_100, "train_100")

#Load model
train_100 = load_model("train_100.sav")
train_100.CheckModel(X_test, 10)
print('du doan dung ', train_100.sroce_pridict(X_train, y_train)*100, '%')