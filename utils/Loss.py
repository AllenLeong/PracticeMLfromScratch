import numpy as np


class Entropy():
    def __init__(self): pass

    def loss(self, y_hat):
        y_hat = np.array(y_hat)

        entropy = np.sum([
                -p*np.log2(p) for p in [
                    (y_hat==i).astype(np.int64).mean() for i in
                        np.unique(y_hat)]
                    ])

        return entropy

    def gradient(self, y_hat):
        y_hat = np.array(y_hat)

        gradient = np.sum([
                (np.log(p)+1)/np.log(2) for p in [
                    (y_hat==i).astype(np.int64).mean() for i in
                        np.unique(y_hat)]
                    ])

        return gradient

class Gini():
    def __init__(self): pass

    def loss(self, y_hat):
        y_hat = np.array(y_hat)

        entropy = np.sum([
                -np.log2(p) for p in [
                    (y_hat==i).astype(np.int64).mean() for i in
                        np.unique(y_hat)]
                    ])

        return entropy

    def gradient(self, y_hat):
        y_hat = np.array(y_hat)

        gradient = np.sum([
                1/np.log(2) for p in [
                    (y_hat==i).astype(np.int64).mean() for i in
                        np.unique(y_hat)]
                    ])

        return gradient

class MeanSquareError():
    def __init__(self): pass

    def loss(self, y, y_hat):
        return np.mean((y-y_hat)**2)/2

    def gradient(self,y, y_hat):
        return np.mean(-(y-y_hat))
