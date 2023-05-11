import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from Composition.SequentialSystem import SequentialSystem
from botorch.acquisition import UpperConfidenceBound
import numpy as np

dtype = torch.float64

def BO_skeleton(system: SequentialSystem, objective="system"):
    parameter_trials = torch.tensor((np.array(system.get_parameters())).flatten(), dtype=dtype).unsqueeze(0)

    if objective=="system":
        def objective():
            return - system.compute_system_loss()
    if objective=="all":
        def objective():
            return - system.compute_system_loss() - sum(system.compute_local_loss())

    result = torch.tensor([objective()], dtype=dtype).reshape([1, 1])
    best_param = None
    best_loss_tuples = None
    best_objective = -10000
    for i in range(30):
        gp = SingleTaskGP(parameter_trials, result)
        UCB = UpperConfidenceBound(gp, beta=0.1)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        bounds = torch.stack([torch.ones(parameter_trials.shape[-1])*0, 2*torch.ones(parameter_trials.shape[-1])])
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )
        parameter_trials = (torch.vstack((parameter_trials, candidate)))
        system.assign_parameters(candidate)
        system_loss = system.compute_system_loss()
        local_loss = system.compute_local_loss() # this should be one loss per component (find a way to index it)
        if objective() > best_objective:
            best_objective = objective()
            best_param = system.get_parameters()
            best_loss_tuples = local_loss, system_loss

        result = torch.vstack((result, torch.tensor([objective()], dtype=dtype).reshape([1, 1])))
    return parameter_trials, best_loss_tuples, best_param
