import torch
from torch.special import *
from Models.Model import Model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# output between (0,1)
class ModelLogistic(Model):
    def __init__(self, inputs, oracle_mode=False, lr=0.001, tol = 1e-05, dtype=torch.float64):
        self.oracle_mode = oracle_mode
        super().__init__(inputs=inputs, lr=lr, tol=tol, dtype = dtype)
        self.model = torch.nn.Linear(inputs, 1)
        self.set_params([1.0]*inputs)
    
    def get_local_loss(self):
        self.model.eval()
        loss = 0

        data, target = self.X, self.y
        
        # if torch.cuda.is_available():
        #     data = data.cuda()
        #     target = target.cuda()
        
        output = self.evaluate(data)
        
        criterion = nn.BCELoss()
        log_loss = criterion(output, target)
        loss += log_loss

        return loss

    def evaluate(self, x, *theta):
        if self.oracle_mode:
            return x # input will be just the label
        with torch.no_grad():
            self.model.weight.copy_(torch.Tensor(list(theta)))
        result = torch.sigmoid(self.model(x))
        return result
    
    def evaluate(self, x):
        if self.oracle_mode:
            return x # input will be just the label
        result = torch.sigmoid(self.model(x))
        return result

    def do_one_descent_on_local(self):
        optimizer = optim.Adam(params=self.parameters(), lr=0.003)
        
        data, target = self.X, self.y
        output = self.evaluate(data)
        loss = nn.BCELoss()(output, target)
        loss.backward()
        optimizer.step()
                
    def descent_to_target_loss(self, target_loss):
        def my_loss(output, target, target_loss):
            criterion = nn.BCELoss()
            cross_entropy_loss = criterion(output, target)
            return (target_loss - cross_entropy_loss)**2
        
        optimizer = optim.Adam(params=self.parameters(), lr=0.003)
        data, target = self.X, self.y
    
        # if torch.cuda.is_available():
        #     data = data.cuda()
        #     target = target.cuda()
        
        # number of iteration
        for x in range(50):
            optimizer.zero_grad()
            output = self.evaluate(data)
            loss = my_loss(output, target, target_loss)
            loss.backward()
            optimizer.step()
    
    def set_oracle_mode(self, bool_val):
        self.oracle_mode = bool_val
    
    def random_initialize_param(self, seed=None):
        
        