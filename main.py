import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
#from tensorflow.keras.utils import to_categorical


def main():
    load_data()

class Sigmoid():
  def acti(self, s):
    return 1.
  def der(self, x):
    return 1.

def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    #print(y)
    #y=to_categorical(y)
    #print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    print(X_test)
    #print(np.shape(X))
    #print(np.shape(y))
    #print(X)


class Sigmoid():
    def acti(self, s):
        return 1.0 / (1.0 + np.exp(-s))

    def der(self, x):
        fx = self.acti(x)
        return fx * (1 - fx)

class Perceptron():
  def __init__(self, input, acti, eta):
    self.W=np.random.rand(input)
    self.Wb=np.random.rand(1)[0]
    self.acti=acti
    self.eta=eta

  def predict(self, x):
    return 1.

  def fit(self, e):
    return 1.

if __name__ == '__main__':
    main()