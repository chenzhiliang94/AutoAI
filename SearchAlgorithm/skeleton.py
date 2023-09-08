import torch
from botorch.models import SingleTaskGP, HeteroskedasticSingleTaskGP, KroneckerMultiTaskGP, MultiTaskGP, FixedNoiseMultiTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from botorch.optim import optimize_acqf
from Composition.SequentialSystem import SequentialSystem
from botorch.acquisition import UpperConfidenceBound
from SearchAlgorithm.AcquisitionFunction import *
import numpy as np
import time

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

def BO_graph(system : DirectedFunctionalGraph, printout=True, iteration=50):
    for node in system.nodes:
        if "Blackbox" in str(node):
            continue
    parameter_trials = torch.tensor((np.array(system.get_all_params()[1])).flatten(), dtype=dtype).unsqueeze(0)

    next_param = torch.DoubleTensor([system.get_all_params()[1]])
    target = []
    input_param = next_param
    best_param = None
    best_objective = -10000
    all_best_losses = []
    for i in range(iteration):
        now = time.time()
        if printout:
            print("BO iteration: ", i)
            print("Current best objective: ", best_objective)
        system.assign_params(next_param[0])
        current_loss = - system.get_system_loss()
        if current_loss > best_objective:
            best_objective = current_loss
            best_param = system.get_all_params()[1]
        all_best_losses.append(best_objective)
        
        UCB = None
        gp = None

        target.append([current_loss])
        Y = torch.DoubleTensor(target)

        # parameterize standardization and model
        # target = standardize(target)
        gp = SingleTaskGP(input_param, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except:
            print("MODEL FITTING, THROW")
            raise Exception("model fitting error")
        # attempts = 0
        # while attempts < 5:
        #     try:
        #         fit_gpytorch_mll(mll)
        #         break
        #     except:
        #         attempts +=1
        UCB = UpperConfidenceBound(gp, beta=1)

        bounds = torch.stack([torch.ones(parameter_trials.shape[-1]) * -1, torch.ones(parameter_trials.shape[-1])])
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=1, raw_samples=20,
        )
        next_param = candidate
        input_param = torch.cat((input_param, next_param), 0)
        later = time.time()
        difference = (later - now)
        print("time taken for one BO iteration: ", difference)
    return all_best_losses, input_param, best_param


def BO_graph_local_loss(system : DirectedFunctionalGraph, bounds: torch.tensor, method, samples, printout=True, iteration=50):
    all_params = []
    for node in system.nodes:
        if "Dummy" in str(node) or "Blackbox" in str(node):
            continue
    
    next_param = system.get_local_losses()
    target = [[-system.get_system_loss()]]

    input_param = next_param
    best_param = None
    best_objective = -10000
    all_best_losses = []
    for i in range(iteration):
        now = time.time()
        if printout:
            print("BO iteration: ", i)
            print("Current best objective: ", best_objective)

        ###
        # assign param which reverse maps to local loss
        current_loss = - system.get_system_loss()
        if i != 0:
            current_loss = - system.reverse_local_loss_lookup(next_param[0], method, samples)
        ###

        # current_loss = - system.get_system_loss()
        if current_loss > best_objective:
            best_objective = current_loss
            best_param = system.get_all_params()[1] 
        all_best_losses.append(best_objective)
        
        UCB = None
        gp = None

        input_param = torch.cat((input_param, system.get_local_losses()), 0)
        target.append([current_loss])
        Y = torch.DoubleTensor(target)

        # parameterize standardization and model
        # target = standardize(target)

        gp = SingleTaskGP(input_param.double(), Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except:
            print("MODEL FITTING, THROW")
        UCB = UpperConfidenceBound(gp, beta=0.1)

        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=1, raw_samples=20,
        )
        next_param = candidate
        if printout:
            print("candidate param: ", next_param)
        later = time.time()
        difference = (later - now)
        print("time taken for one BO iteration: ", str(difference))
    return all_best_losses, best_param