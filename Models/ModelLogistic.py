import torch
from torch.special import *
from Models.Model import Model

INPUTS=2

class ModelLogistic(Model):
    def __init__(self, theta_0_=1., theta_1_=1., lr=0.01, tol = 1e-05, dtype=torch.float64):
        super().__init__(inputs=INPUTS, lr=lr, tol=tol, dtype = dtype)
        self.set_params([theta_0_,theta_1_])

    def evaluate(self, x, theta_0_, theta_1_):
        return expit(theta_0_ * x[0] + theta_1_ *  x[1])