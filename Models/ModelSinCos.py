from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
import math

class sincosModel(BaseEstimator):
    theta_0_ = 1
    theta_1_ = 1

    lr = 0.01
    tol = 1e-05

    def __init__(self, lr=0.01):
        self.lr = lr

    def get_params(self):
        return [self.theta_0_, self.theta_1_]

    def func(self, x):
        return self.theta_0_ * np.sin(self.theta_1_ * x + 0.5)

    def func_all(self, X, theta_0, theta_1):
        return theta_0 * np.sin(theta_1 * X + 0.5)

    def get_gradients_default(self, x):
        # dy/d_theta with theta from class attribute
        return [np.sin(self.theta_1_ * x + 0.5), self.theta_0_ * np.cos(self.theta_1_ * x + 0.5) * x]

    def get_gradients_combined(self, x, theta):
        theta_0, theta_1 = theta[0], theta[1]
        # dy/d_theta
        return [np.sin(theta_1 * x + 0.5), theta_0 * np.cos(theta_1 * x + 0.5) * x]

    def get_gradients(self, x, theta_0, theta_1):
        # dy/d_theta
        return np.sin(theta_1 * x + 0.5), theta_0 * np.cos(theta_1 * x + 0.5) * x

    def set_theta(self, theta):
        self.theta_0_ = theta[0]
        self.theta_1_ = theta[1]

    def fit(self, X, y):
        all_theta = []
        itr_max = 50
        itr = 0
        while (itr < itr_max):
            gradient_theta_0 = 0
            gradient_theta_1 = 0
            idx = np.random.choice(np.arange(len(X)), 100, replace=False)

            for x, y_truth in zip(X[idx], np.array(y)[idx]):
                difference = (self.theta_0_ * np.sin(self.theta_1_ * x + 0.5) - y_truth)
                gradient_theta_0 += np.sin(self.theta_1_ * x + 0.5) * difference
                gradient_theta_1 += self.theta_0_ * np.cos(self.theta_1_ * x + 0.5) * x * difference

            gradient_theta_0 /= len(idx)
            gradient_theta_1 /= len(idx)


            if np.abs(gradient_theta_0) <= self.tol and np.abs(gradient_theta_1) <= self.tol:
                break

            self.theta_0_ -= self.lr * gradient_theta_0
            self.theta_1_ -= self.lr * gradient_theta_1
            all_theta.append((self.theta_0_, self.theta_1_))
            itr += 1
        return all_theta
    
    def predict(self, X):
        #check_is_fitted(self)
        X = check_array(X)
        f = np.vectorize(self.func)
        output = f(np.array(X))
        return output

def sincosModelGroundTruth(X, theta_0, theta_1):
    y = []
    for x in X:
        y.append(theta_0 * np.sin(theta_1 * x + 0.5) + np.random.normal(0, 0.05))
    return y


