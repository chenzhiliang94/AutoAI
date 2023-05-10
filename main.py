import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from Components.ConditionalNormalDistribution import conditional1DNormal
from Components.DifferentiablePolynomial import DifferentiablePolynomial
from Models.ModelExponential import ExponentialModel
from Plotting.HeatMapLossFunction import *
from Composition.SequentialSystem import SequentialSystem
from Search.skeleton import BO_skeleton

# f = conditional1DNormal()
f = DifferentiablePolynomial() # black box function (comes second)
f.noisy_operation = lambda y, n:(1+n)*y # set multiplicative noise
g = ExponentialModel() # white box function (comes first)

ground_truth_theta_0 = 0.6
ground_truth_theta_1 = 0.2

# generate local dataset over g
X_local = torch.tensor(np.random.uniform(1, 3, size=100))
y_local = g(X_local, params=[ground_truth_theta_0, ground_truth_theta_1], noisy = True) # labeling effort of local
# generate end to end dataset (use same X)
X_global = X_local
z_global = f(y_local, noisy = False) # ground truth
z_global_pertubed = f(y_local, noisy = True)
plt, fig, ax = HeatMapLossFunction(X_local, y_local, X_global, z_global_pertubed, f, g, plt)

# train function g on x,y ("local") using local gradient descent
all_theta_via_local = g.fit(X_local,y_local)
y_pred = g(X_global.reshape(len(X_global),1))

# train composite function f.g on x,z ("global") using BO
s = SequentialSystem()
g = ExponentialModel() # reset
s.addModel(g, X_local, y_local)
s.addComponent(f)
s.addGlobalData(X_global, z_global)

all_theta_via_global = BO_skeleton(s)
#all_theta_via_global = s.fit_global_differentiable() # this performs end to end gradient descent

all_theta_via_local = np.array(all_theta_via_local)
all_theta_via_global = np.array(all_theta_via_global)

ax[0].scatter(all_theta_via_local[:,0], all_theta_via_local[:,1], s=0.2,alpha=1, label="gradient climbing over local data set")
ax[0].scatter(all_theta_via_global[:,0], all_theta_via_global[:,1], s=1,alpha=1, label="end to end learning")
lgnd = ax[0].legend()

ax[1].scatter(all_theta_via_local[:,0], all_theta_via_local[:,1], s=0.2,alpha=0.5)
ax[1].scatter(all_theta_via_global[:,0], all_theta_via_global[:,1], s=1,alpha=0.5)

plt.show()


# prediction_y = model(X_local.reshape(len(X_local),1))
# prediction_z_correct, prediction_z_pertubed = component.generate_both(prediction_y) # component is not perfect
#
# print("local loss: ", mean_squared_error(model(X_local.reshape(len(X_local),1)), y_local))
# print("system loss with erronous component: ", mean_squared_error(prediction_z_pertubed, z_local_ground_truth))
# print("system loss with correct component: ", mean_squared_error(prediction_z_correct, z_local_ground_truth))
