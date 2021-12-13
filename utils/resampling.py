import numpy as np

class Bootstrap():
    """
    Class Bootstrap Generator
    """
    def __init__(self, X,y=None,n = 10, random_state=None):
        self.X = X
        self.y = y
        self.n = n
        self.random_state = random_state

    def __iter__(self):
        np.random.seed(self.random_state)
        self.n_sample =  0
        self.random_states = np.random.randint(1,self.X.shape[0],self.n)
        return self

    def __next__(self):
        if self.n_sample  == self.n:
            raise StopIteration
        else:
            random_state = self.random_states[self.n_sample]
            np.random.seed(random_state)
            idx = np.random.choice(range(self.X.shape[0]), self.X.shape[0])
            self.n_sample += 1
            if self.y is None:
                return self.X[idx]
            else:
                return self.X[idx], self.y[idx]
