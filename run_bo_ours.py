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
import botorch
from numpy import genfromtxt

botorch.settings.debug = False

import random 

# def run_experiments_all(system_param : dict, data_generation_seed : int, vanilla_bo_trials : int, vanilla_bo_iterations: int, vanilla_bo_output_dir, vanilla_bo_all_seeds_dir,
#                         our_bo_trial : int, ours_bo_bound_size: int, our_bo_iterations : int, our_bo_samples: list, our_bo_output_dir, our_bo_search_method):
#     DG = generate_dg(system_param, data_generation_seed)
#     nx.draw_networkx(DG)

#     #grad descent
#     DG.random_initialize_param(data_generation_seed)
#     lower_bound_local_loss, all_loss = show_system_loss_from_grad_descent(DG, itr=500, plot=True)
#     lower_bound_local_loss = [x.detach().numpy().tolist() for x in lower_bound_local_loss]

#     # vanilla BO

#     vanilla_all_trials = []

#     trials = 100
#     seeds = []    
#     for x in range(trials):
#         print("trial of vanilla BO: ", x)
#         r1 = random.randint(0, 100000)
#         if len(vanilla_all_trials) > vanilla_bo_trials:
#             break
#         try:
#             DG.random_initialize_param(r1)
#             DG.fit_locally_partial(100)
#             if (DG.get_system_loss() > 100):
#                 continue
#             all_best_losses, _, _ = BO_graph(DG,printout=True,iteration=vanilla_bo_iterations)
#             vanilla_all_trials.append(all_best_losses)
#             seeds.append(r1)
#         except:
#             print("exception in vanilla BO")
#             continue

#     vanilla_all_trials = np.array(vanilla_all_trials)
#     np.savetxt(vanilla_bo_output_dir, vanilla_all_trials)
#     print("seeds: ")
#     print(seeds)
#     np.savetxt(vanilla_bo_all_seeds_dir, seeds)
#     run_our_bo(DG, lower_bound_local_loss, vanilla_bo_all_seeds_dir,  our_bo_trial, ours_bo_bound_size, our_bo_iterations, our_bo_samples, our_bo_output_dir, our_bo_search_method)

# def run_our_bo(DG : DirectedFunctionalGraph, lower_bound_local_loss, seed_dir,  our_bo_trial : int, ours_bo_bound_size: int, our_bo_iterations : int, our_bo_samples: list, our_bo_output_dir, our_bo_search_method):
#     seeds = genfromtxt(seed_dir, delimiter=' ')
#     samples = our_bo_samples
#     for s in samples:
#         print("samples: ", s)
#         loss_space_bo_all_trials = []
#         for x in range(200):
#             print("number of attempts: ", x)
#             print("trial of our BO (successful): ", len(loss_space_bo_all_trials))
#             if len(loss_space_bo_all_trials) > our_bo_trial:
#                 break
#             try:
#                 DG.random_initialize_param(int(seeds[0]))
#                 DG.fit_locally_partial(100)
#                 # BO with local loss -> system loss
#                 print("bounds: ", np.array(lower_bound_local_loss))
#                 bounds = torch.tensor([np.array(lower_bound_local_loss), np.array(lower_bound_local_loss) * ours_bo_bound_size])
#                 all_best_losses_ours, best_param = BO_graph_local_loss(DG, bounds, our_bo_search_method, s, printout=True, iteration=our_bo_iterations)
#                 loss_space_bo_all_trials.append(all_best_losses_ours)
#                 seeds.pop(0)
#             except:
#                 continue

#         loss_space_bo_all_trials = np.array(loss_space_bo_all_trials)
#         file_name = our_bo_output_dir + "_" + str(s) + ".csv"
#         np.savetxt(file_name, loss_space_bo_all_trials)

ground_truth_param = {1 : np.array([0.7, 0.9, -0.5]), 2: np.array([0.3, 0.7]), 8: np.array([-0.2, -0.2]), 9: np.array([-0.5, 0.5]), 10: np.array([-0.2, 0.1]), 11: np.array([-0.3, 0.3, 0.2]),
                      12: np.array([0.1, 0.1]), "Blackbox3":np.array([1.2, 0.8]), 4:np.array([1.1, -0.5]), "Blackbox5":np.array([0.7, -0.5]),
                      "Blackbox6": np.array([0.7, 1.1]), 7: np.array([0.7, -0.5])}

data_generation_seed = 5
vanilla_trials = 3
vanilla_iterations = 100
vanilla_dir = "result/vanilla_BO_300_equal_queries.csv"
vanilla_bo_all_seeds_dir = "result/seeds_runD_equal_queries_seed.csv"
our_bo_trial = 3
ours_bo_bound_size = 15
our_bo_iterations = [300,60,30,15,10]
our_bo_samples = [1,5,10,20,30]
our_bo_output_dir = "result/our_bo_300_equal_queries"
our_bo_search_method = "multi_search"
run_experiments_all_same_queries(ground_truth_param,data_generation_seed, vanilla_trials, vanilla_iterations, vanilla_dir, vanilla_bo_all_seeds_dir,
                        our_bo_trial, ours_bo_bound_size, our_bo_iterations, our_bo_samples, our_bo_output_dir, our_bo_search_method)

