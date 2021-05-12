import numpy as np

def load_data_split(fn="data", dirname="./data/"):
    X = np.loadtxt(dirname+fn+"-x.csv", delimiter=',', dtype='str', encoding='utf-8-sig')
    Y = np.loadtxt(dirname+fn+"-y.csv", delimiter=',', encoding='utf-8-sig')
    X_train = X[:int(X.shape[0] * 0.8)]
    X_test = X[int(X.shape[0] * 0.8):]
    Y_train = Y[:int(Y.shape[0] * 0.8)]
    Y_test = Y[int(Y.shape[0] * 0.8):]
    return X_train, X_test, Y_train, Y_test