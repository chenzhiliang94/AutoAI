import torch
from botorch.models import SingleTaskGP, HeteroskedasticSingleTaskGP, KroneckerMultiTaskGP, MultiTaskGP, FixedNoiseMultiTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from botorch.optim import optimize_acqf
from Composition.SequentialSystem import SequentialSystem
from botorch.acquisition import UpperConfidenceBound
from Search.AcquisitionFunction import *
import numpy as np

from GraphDecomposition.DirectedFunctionalGraph import *

dtype = torch.float64

def BO_skeleton(system: SequentialSystem, objective="system", model="single_task_gp", printout=False):
    parameter_trials = torch.tensor((np.array(system.get_parameters())).flatten(), dtype=dtype).unsqueeze(0)

    next_param = torch.DoubleTensor([[1.0, 1.0, 1.0, 1.0]])
    target = []
    input_param = next_param
    best_param = None
    best_loss_tuples = None
    best_objective = -10000
    for i in range(30):
        if printout:
            print("BO iteration: ", i)
            print("Current best objective: ", best_objective)
        system.assign_parameters(next_param)
        if objective == "all":
            current_loss = - system.compute_system_loss() - sum(system.compute_local_loss())
        if objective == "all":
            current_loss = - system.compute_system_loss()
        if current_loss > best_objective:
            best_objective = current_loss
            best_param = system.get_parameters()
            best_loss_tuples = system.compute_local_loss(), system.compute_system_loss()

        UCB = None
        gp = None
        if model == "single_task_gp":
            target.append([- system.compute_system_loss() - sum(system.compute_local_loss())])
            Y = torch.DoubleTensor(target)

            # parameterize standardization and model
            # target = standardize(target)
            gp = SingleTaskGP(input_param, Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll);
            UCB = UpperConfidenceBound(gp, beta=1)

        if model == "multi_task_gp_bonilla":
            target.append([-system.compute_local_loss()[0], -system.compute_local_loss()[1], -system.compute_system_loss()])
            Y = torch.DoubleTensor(target)

            # parameterize standardization and model
            # target = standardize(target)
            gp = KroneckerMultiTaskGP(input_param, Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll);
            UCB = ScalarizedUpperConfidenceBound(gp, beta=1, weights=torch.tensor([1.0, 1.0, 1.0]).double())

        bounds = torch.stack([torch.ones(parameter_trials.shape[-1])*0, 2*torch.ones(parameter_trials.shape[-1])])
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=1, raw_samples=20,
        )
        next_param = candidate
        input_param = torch.cat((input_param, next_param), 0)
    return parameter_trials, best_loss_tuples, best_param

def BO_graph(system : DirectedFunctionalGraph, printout=True):
    all_params = []
    for node in system.nodes:
        if "Blackbox" in str(node):
            continue
    parameter_trials = torch.tensor((np.array(system.get_all_params()[1])).flatten(), dtype=dtype).unsqueeze(0)

    next_param = torch.DoubleTensor([system.get_all_params()[1]])
    target = []
    input_param = next_param
    best_param = None
    best_objective = -10000
    for i in range(100):
        if printout:
            print("BO iteration: ", i)
            print("Current best objective: ", best_objective)
        system.assign_params(next_param[0])
        current_loss = - system.get_system_loss()
        if current_loss > best_objective:
            best_objective = current_loss
            best_param = system.get_all_params()[1]

        UCB = None
        gp = None

        target.append([current_loss])
        Y = torch.DoubleTensor(target)

        # parameterize standardization and model
        # target = standardize(target)
        gp = SingleTaskGP(input_param, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll);
        UCB = UpperConfidenceBound(gp, beta=1)

        bounds = torch.stack([torch.ones(parameter_trials.shape[-1]) * 0, 2 * torch.ones(parameter_trials.shape[-1])])
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=1, raw_samples=20,
        )
        next_param = candidate
        input_param = torch.cat((input_param, next_param), 0)
    return parameter_trials, best_param


def BO_graph_local_loss(system : DirectedFunctionalGraph, printout=True):
    all_params = []
    for node in system.nodes:
        if "Blackbox" in str(node):
            continue

    next_param = torch.DoubleTensor([system.get_local_losses()])
    target = []
    input_param = next_param
    best_param = None
    best_objective = -10000
    for i in range(100):
        if printout:
            print("BO iteration: ", i)
            print("Current best objective: ", best_objective)

        ###
        # find param which reverse maps to local loss
        something_something = input_param
        system.assign_params(something_something)
        ###

        current_loss = - system.get_system_loss()
        if current_loss > best_objective:
            best_objective = current_loss
            best_param = system.get_all_params()[1]

        UCB = None
        gp = None

        target.append([current_loss])
        Y = torch.DoubleTensor(target)

        # parameterize standardization and model
        # target = standardize(target)
        gp = SingleTaskGP(input_param, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll);
        UCB = UpperConfidenceBound(gp, beta=1)

        bounds = torch.stack([torch.ones(parameter_trials.shape[-1]) * 0, 2 * torch.ones(parameter_trials.shape[-1])])
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=1, raw_samples=20,
        )
        next_param = candidate
        input_param = torch.cat((input_param, next_param), 0)
    return parameter_trials, best_param