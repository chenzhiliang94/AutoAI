import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from collections import defaultdict

from Components.ConditionalNormalDistribution import ConditionalNormalDistribution
from Components.DifferentiablePolynomial import DifferentiablePolynomial
from Models.ModelExponential import ModelExponential
from Models.ModelSinCos import ModelSinCos
from Models.ModelLogistic import ModelLogistic
from Models.ModelSigmoid import ModelSigmoid
from Composition.SequentialSystem import SequentialSystem
from SearchAlgorithm.skeleton import BO_skeleton, BO_graph, BO_graph_local_loss

from GraphDecomposition.DirectedFunctionalGraph import DirectedFunctionalGraph
from Components.DifferentiablePolynomial import DifferentiablePolynomial
from Models.ModelSinCos import ModelSinCos
from Models.ModelConstant import ModelConstant
from Models.ModelWeightedSum import ModelWeightedSum
from Models.ModelExponential import ModelExponential

from GraphDecomposition.Heuristic import *
from helper import *
from Plotting.HeatMapLossFunction import *

from Models.ModelMNIST import ModelMNIST
from mnist.MNISTLoader import *
from helper import *
import time

ground_truth_param_mnist = {"Blackbox7": np.array([0.3,0.45]), 6 : np.array([0.7, 0.9, -0.5]), 2: np.array([0.5, -.7]), 12: np.array([0.1, 0.1]), "Blackbox3":np.array([1.2, 0.8]), 4:np.array([-0.3, 0.5])}
ground_truth_param_mnist = {"Blackbox7": np.array([0.8,0.8]), 6 : np.array([0.7, 0.9, 0.5]), 2: np.array([0.5, 0.7]), 12: np.array([0.1, 0.1]), "Blackbox3":np.array([1.2, 0.8]), 4:np.array([-0.3, 0.5])}

# gradient_system_losses_trials = []
# ours_bo_system_losses_trials = []
# for x in range(100):
#     if len(gradient_system_losses_trials) > trials:
#         break
    
#     try:
#         seed = np.random.randint(1,1000000)
#         #grad descent
#         dg_nn.random_initialize_param(seed)
#         dg_nn.to_perturb = False
#         lower_bound_local_loss, all_loss = show_system_loss_from_grad_descent(dg_nn, itr=500, plot=False)
#         lower_bound_local_loss = [x.detach().cpu().numpy() for x in lower_bound_local_loss]
#         min_system_loss_achieved_by_grad =  min(all_loss["system"])

#         dg_nn.random_initialize_param(seed)
#         # BO with local loss -> system loss

#         bounds = torch.tensor([np.array(lower_bound_local_loss), np.array(lower_bound_local_loss) * 10])
#         all_best_losses_ours, best_param = BO_graph_local_loss(dg_nn, bounds, "nn_lookup", iteration=20)

#         gradient_system_losses_trials.append(min_system_loss_achieved_by_grad)
#         ours_bo_system_losses_trials.append(all_best_losses_ours)
#     except:
#         continue
    



# print(gradient_system_losses_trials)
# gradient_system_losses_trials = np.array(gradient_system_losses_trials)
# ours_bo_system_losses_trials = np.array(ours_bo_system_losses_trials)

# # mean = np.mean(loss_space_bo_all_trials, axis=0)
# # std = np.std(loss_space_bo_all_trials, axis=0)

# # combined_array_ours = np.vstack((mean, std)).T
# np.savetxt("mnist_gradient.csv", gradient_system_losses_trials)
# np.savetxt("mnist_ours_20_iterations.csv", ours_bo_system_losses_trials)
data_generation_seed = 99
dg_nn = create_mnist_system(ground_truth_param_mnist,  noise=0.8, seed=data_generation_seed)
nx.draw_networkx(dg_nn)

our_bo_trial = 5
ours_bo_bound_size = 5
our_bo_iterations = 10
our_bo_samples = [10]
our_bo_output_dir = "result/mnist_bo_10_800_99_new"
our_bo_search_method = "nn_lookup"

vanilla_bo_trials = 5
vanilla_bo_iterations = 100
vanilla_bo_output_dir = "result/mnist_vanilla_100_800_99_new.csv"
    

seed = 800
seeds = [seed]

gradient_system_losses_trials = []
lower_bound_local_loss = None
for s in seeds:
    now = time.time()
    print("doing grad descent")
    dg_nn.random_initialize_param(s)
    lower_bound_local_loss, all_loss = show_system_loss_from_grad_descent(dg_nn, itr=300, plot=False)
    lower_bound_local_loss = [x.detach().cpu().numpy() for x in lower_bound_local_loss]
    gradient_system_losses_trials.append(min(all_loss["system"]))
    later = time.time()
    print("time taken: ", later - now)

np.savetxt("result/mnist_gradient.csv", gradient_system_losses_trials)

# vanilla BO - but ignore nn components

vanilla_all_trials = []

trials = 100
seeds = []    
for x in range(trials):
    print("trial of vanilla BO: ", x)
    r1 = seed
    if len(vanilla_all_trials) >= vanilla_bo_trials:
        break
    try:
        dg_nn.random_initialize_param(r1)
        if (dg_nn.get_system_loss() > 120):
            print(dg_nn.get_system_loss(), ": seed is bad")
            continue
        for x in range(200):
            dg_nn.nodes["nn_1"]["component"].do_one_descent_on_local()
            dg_nn.nodes["nn_5"]["component"].do_one_descent_on_local()
        
        all_best_losses, _, _ = BO_graph(dg_nn,printout=True,iteration=vanilla_bo_iterations)
        vanilla_all_trials.append(all_best_losses)
        seeds.append(r1)
    except:
        print("exception in vanilla BO")
        continue

vanilla_all_trials = np.array(vanilla_all_trials)
np.savetxt(vanilla_bo_output_dir, vanilla_all_trials)
    

run_our_bo(dg_nn, lower_bound_local_loss, seeds,  our_bo_trial, ours_bo_bound_size, our_bo_iterations, our_bo_samples, our_bo_output_dir, our_bo_search_method)