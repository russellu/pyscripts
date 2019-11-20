import numpy as np
from sklearn.datasets import load_linnerud

class GNBModel:
    def GuassianNB(self, x, mean, variance):
        return np.exp(-np.square(x - mean) / (2. * variance) )/ np.sqrt(2. * np.pi * variance)

    def fit(self, x_, y_):
        x_num = len(x_)
        if len(x_.shape)>1:
            x_feature_num = x_.shape[1]
        else:
            x_feature_num = 1
        y_set = np.unique(y_)
        y_num = len(y_set)
        self.y_num = y_num
        self.y_prob = np.zeros([y_num])
        self.mean = np.zeros([y_num, x_feature_num])
        self.variance = np.zeros([y_num, x_feature_num])
        for y_value in y_set:
            int_y_value=int(y_value)
            self.y_prob[int_y_value] = np.sum(y_==y_value) / x_num
            self.mean[int_y_value] = np.mean(x_[y_==y_value], axis=0)
            self.variance[int_y_value] = np.var(x_[y_==y_value], axis=0)

        return self

    def probyx(self, x_,y_value):
        x_num = len(x_)
        px = np.zeros([x_num])
        for x_values in range(x_num):
            mean = self.mean[y_value]
            var = self.variance[y_value]
            px[x_values] = self.y_prob[y_value] * np.prod(self.GuassianNB(x_[x_values], mean, var))
        return px


