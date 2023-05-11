import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from Components.ConditionalNormalDistribution import conditional1DNormal
from Components.DifferentiablePolynomial import DifferentiablePolynomial
from Models.ModelExponential import ExponentialModel
from Models.ModelSinCos import sincosModel
from Plotting.HeatMapLossFunction import *
from Composition.SequentialSystem import SequentialSystem
from Search.skeleton import BO_skeleton

# f = conditional1DNormal()
f = DifferentiablePolynomial() # black box function (comes second)
f.noisy_operation = lambda y, n:(1+n)*y # set multiplicative noise
g_A = ExponentialModel() # white box function (comes first)
g_B = sincosModel() # white box function (comes last)

ground_truth_theta_0_A = 1
ground_truth_theta_1_A = 1

ground_truth_theta_0_B = 0.9
ground_truth_theta_1_B = 1.4

# generate local dataset over g (first component A)
X_local = torch.tensor(np.random.uniform(1, 3, size=100))
X_global = X_local
y_local = g_A(X_local, params=[ground_truth_theta_0_A, ground_truth_theta_1_A], noisy = True) # labeling effort of local
# pass into black box component
X_b = f(y_local, noisy = False) # ground truth
X_b_pertubed = f(y_local, noisy = True)
# generate local dataset over g2 (second component B)
system_output_no_perturbation = g_B(X_b, params=[ground_truth_theta_0_B, ground_truth_theta_1_B], noisy = True) # labeling effor
system_output_perturbation = g_B(X_b_pertubed, params=[ground_truth_theta_0_B, ground_truth_theta_1_B], noisy = True) # labeling effor

# generate end to end dataset (use same X)
X_global = X_local
z_global = system_output_no_perturbation
z_global_pertubed = system_output_perturbation


#plt, fig, ax = HeatMapLossFunction(X_local, y_local, X_global, z_global_pertubed, f, g, plt)

# train function g on x,y ("local") using local gradient descent
#all_theta_via_local = g_A.fit(X_local,y_local)
#y_pred = g_A(X_global.reshape(len(X_global),1))

# train composite function f.g on x,z ("global") using BO
s = SequentialSystem()

s.addModel(g_A, X_local, y_local)
s.addComponent(f)
s.addModel(g_B, X_b, system_output_no_perturbation)
s.addGlobalData(X_global, z_global_pertubed)

all_theta_via_global, loss, param = BO_skeleton(s, objective="all")
print("local, system loss (best)")
print(loss)
print("param")
print(param)
#all_theta_via_global = s.fit_global_differentiable() # this performs end to end gradient descent
#
# all_theta_via_local = np.array(all_theta_via_local)
# all_theta_via_global = np.array(all_theta_via_global)
#
# ax[0].scatter(all_theta_via_local[:,0], all_theta_via_local[:,1], s=0.2,alpha=1, label="gradient climbing over local data set")
# ax[0].scatter(all_theta_via_global[:,0], all_theta_via_global[:,1], s=1,alpha=1, label="end to end learning")
# lgnd = ax[0].legend()
#
# ax[1].scatter(all_theta_via_local[:,0], all_theta_via_local[:,1], s=0.2,alpha=0.5)
# ax[1].scatter(all_theta_via_global[:,0], all_theta_via_global[:,1], s=1,alpha=0.5)
#
# plt.show()


# prediction_y = model(X_local.reshape(len(X_local),1))
# prediction_z_correct, prediction_z_pertubed = component.generate_both(prediction_y) # component is not perfect
#
# print("local loss: ", mean_squared_error(model(X_local.reshape(len(X_local),1)), y_local))
# print("system loss with erronous component: ", mean_squared_error(prediction_z_pertubed, z_local_ground_truth))
# print("system loss with correct component: ", mean_squared_error(prediction_z_correct, z_local_ground_truth))
