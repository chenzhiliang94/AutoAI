# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
#
# from Components.ConditionalNormalDistribution import ConditionalNormalDistribution
# from Components.DifferentiablePolynomial import DifferentiablePolynomial
# from Models.ModelExponential import ModelExponential
# from Models.ModelSinCos import ModelSinCos
# from Plotting.HeatMapLossFunction import *
# from Composition.SequentialSystem import SequentialSystem
# from Search.skeleton import BO_skeleton
#
# # f = ConditionalNormalDistribution()
# f = DifferentiablePolynomial() # black box function (comes second)
# f.noisy_operation = lambda y, n:(1+n)*y # set multiplicative noise
# g_A = ModelExponential() # white box function (comes first)
# g_A.noisy_operation = lambda y, n:y+n # set addition noise
# g_B = ModelSinCos() # white box function (comes last)
# g_B.noisy_operation = lambda y, n:y+n # set addition noise
#
# ground_truth_theta_0_A = 1.6
# ground_truth_theta_1_A = 1.2
#
# ground_truth_theta_0_B = 0.9
# ground_truth_theta_1_B = 1.4
#
# # generate local dataset over g (first component A)
# X_local = torch.tensor(np.random.uniform(1, 3, size=100))
# X_global = X_local
# y_local = g_A(X_local, params=[ground_truth_theta_0_A, ground_truth_theta_1_A], noisy = True, noise_mean = 0.0) # labeling effort of local
# # pass into black box component
# X_b = f(y_local, noisy = False) # ground truth, no noise in black box component
# # generate local dataset over g2 (second component B)
# system_output_no_perturbation = g_B(X_b, params=[ground_truth_theta_0_B, ground_truth_theta_1_B], noisy = True, noise_mean = 0.0) # labeling effort
#
# # generate end to end dataset (use same X)
# X_global = X_local
# z_global = system_output_no_perturbation
#
# # local gradient descent
# all_theta_via_local = g_A.fit(X_local,y_local)
# all_theta_via_local = g_B.fit(X_b,system_output_no_perturbation)
#
#
# # create the system
# s = SequentialSystem()
#
# s.addModel(g_A, X_local, y_local)
# s.addComponent(f)
# s.addModel(g_B, X_b, system_output_no_perturbation)
# s.addGlobalData(X_global, z_global)
#
# # show parameters and local losses (components already have converged params from gradient descent)
# print(s.get_parameters())
# print(s.compute_local_loss())
# print(s.compute_system_loss())
#
# # BO - parameters are resetted
# all_theta_via_global, loss, param = BO_skeleton(s, objective="all", model="single_task_gp", printout=True)
# print("local, system loss (best)")
# print(loss)
# print("param")
# print(param)
#
# #plt, fig, ax = HeatMapLossFunction(X_local, y_local, X_global, z_global_pertubed, f, g, plt)
#
# #all_theta_via_global = s.fit_global_differentiable() # this performs end to end gradient descent
# #
# # all_theta_via_local = np.array(all_theta_via_local)
# # all_theta_via_global = np.array(all_theta_via_global)
# #
# # ax[0].scatter(all_theta_via_local[:,0], all_theta_via_local[:,1], s=0.2,alpha=1, label="gradient climbing over local data set")
# # ax[0].scatter(all_theta_via_global[:,0], all_theta_via_global[:,1], s=1,alpha=1, label="end to end learning")
# # lgnd = ax[0].legend()
# #
# # ax[1].scatter(all_theta_via_local[:,0], all_theta_via_local[:,1], s=0.2,alpha=0.5)
# # ax[1].scatter(all_theta_via_global[:,0], all_theta_via_global[:,1], s=1,alpha=0.5)
# #
# # plt.show()
#
#
# # prediction_y = model(X_local.reshape(len(X_local),1))
# # prediction_z_correct, prediction_z_pertubed = component.generate_both(prediction_y) # component is not perfect
# #
# # print("local loss: ", mean_squared_error(model(X_local.reshape(len(X_local),1)), y_local))
# # print("system loss with erronous component: ", mean_squared_error(prediction_z_pertubed, z_local_ground_truth))
# # print("system loss with correct component: ", mean_squared_error(prediction_z_correct, z_local_ground_truth))


from GraphDecomposition.Heuristic import *
from GraphDecomposition.DirectedFunctionalGraph import *
from Components.DifferentiablePolynomial import *
from Models.ModelSinCos import *
from Models.ModelWeightedSum import *
from Models.ModelExponential import *

G = DirectedFunctionalGraph()

G = nx.DiGraph()
G.add_node(0, idx="1", component=ModelWeightedSum())
G.add_node(1, idx="1", component=DifferentiablePolynomial())
G.add_node(2, idx="2", component=ModelSinCos())
G.add_node("BlackboxB", idx="3",  component=ModelWeightedSum())
G.add_node(8, idx="8", component=DifferentiablePolynomial())
G.add_node(4, idx="4", component=ModelSinCos())
G.add_node(5, idx="5", component=ModelWeightedSum())
G.add_node(7, idx="7", component=DifferentiablePolynomial())
G.add_node("exit", idx="8", component=ModelWeightedSum())
G.add_node("BlackboxA", idx="6",  component=DifferentiablePolynomial())
G.add_node(6, idx="6", component=ModelWeightedSum())

G.add_edge(0, 2)
G.add_edge(1, 2)
G.add_edge(1, 5)
G.add_edge(2, "BlackboxB")
G.add_edge("BlackboxB",8)
G.add_edge(8,7)
G.add_edge(5, "BlackboxA")
G.add_edge("BlackboxB", 4)
G.add_edge("BlackboxA", 4)
G.add_edge(4, 7)
G.add_edge(7, "exit")
G.add_edge(6, 5)

all_black_box = ["BlackboxA", "BlackboxB"]
all_decomp = find_all_decomposition_full(all_black_box, G)
all_valid_decomp = get_all_valid_decomposition(all_decomp)
l = 1
best_decomposition, score = get_best_decomposition(all_valid_decomp, G, l=1)
print("\n", "Best decomposition:")
print(goodness_measure(G, best_decomposition, l))

plot(G)
plt.show()
