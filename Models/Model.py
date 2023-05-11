import torch
import torch.nn as nn
import torch.optim as optim

def islist(obj):
    return isinstance(obj, list)

def recursive_map(fn, iterable):
    return list(map(lambda i: recursive_map(fn, i) if islist(i) else fn(i), iterable))

class Model(nn.Module):
    params = None
    X = None
    y = None
    def __init__(self, lr=0.01, tol = 1e-05, dtype=torch.float64):
        super().__init__()
        self.lr = lr
        self.tol = tol
        self.dtype = dtype
        self.noisy_operation = (lambda y, n : y + n)

    def attach_local_data(self, X, y):
        self.X = X
        self.y = y

    def get_local_loss(self):
        assert self.X != None and self.y != None, "no local data is found. Cannot get local loss!"
        output = self(self.X, grad=True)
        mse = nn.MSELoss()
        loss = mse(output, self.y)
        return loss

    def to_param(self, params):
        return recursive_map(lambda p: nn.Parameter(p if torch.is_tensor(p) else torch.tensor(p, dtype=self.dtype)), params)

    def set_params(self, params):
        self.params = self.to_param(params)
        self.params_list = nn.ParameterList(self.params)

    def get_params(self):
        return recursive_map(lambda p: p.item(), self.params)

    def get_gradients_default(self, x):
        # dy/d_theta with theta from class attribute
        return self.get_gradients(x, *self.params)

    def get_gradients_combined(self, x, theta):
        # dy/d_theta
        return self.get_gradients(x, *theta)

    def get_gradients(self, x, params = None):
        if not params is None:
            assert self.check_param(params)
            params = self.to_param(params)
        else:
            params = self.params
        recursive_map(lambda p: p.grad.zero_() if p.grad else None, params)
        y = self.evaluate(x, *params)
        y.backward()
        g = recursive_map(lambda p: p.grad, params)
        return g

    def fit(self, X, y, itr_max = 1000):
        optimizer = optim.Adam(self.params, lr = self.lr)
        all_theta = [self.get_params()]
        mse = nn.MSELoss()
        for _ in range(itr_max):
            optimizer.zero_grad()
            output = self(X, grad=True)
            loss = mse(output, y)
            loss.backward()
            optimizer.step()
            all_theta.append(self.get_params())
            if loss < self.tol:
                break
        return all_theta

    # Recursively check that the supplied params is the same length as the original params
    def check_param(self, params):
        def recursive_check(self_param, other_param):
            self_islist = islist(self_param)
            other_islist = islist(other_param)
            if self_islist != other_islist:
                return False
            if not self_islist and not other_islist:
                return True
            if len(self_param) != len(other_param):
                return False
            for s_p, o_p in zip(self_param, other_param):
                if not recursive_check(s_p, o_p):
                    return False
            return True
        return recursive_check(self.params, params)

    def forward(self, X, params = None, grad = False, noisy=False, noise_mean = 0.3, noise_std = 0.05, noisy_operation = None):
        if params is None:
            params = self.params
        else:
            assert self.check_param(params)  # If params is supplied, must be equal length to number of params
            params = self.to_param(params)
        y = self.evaluate(X, *params)
        if noisy:
            noise = torch.normal(torch.full_like(y,noise_mean), torch.full_like(y,noise_std))
            if noisy_operation is None:
                noisy_operation = self.noisy_operation
            y = noisy_operation(y, noise)
        if not grad:
            y = y.detach()
        return y