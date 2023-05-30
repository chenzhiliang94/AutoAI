from GraphDecomposition.Heuristic import *
from GraphDecomposition.DirectedFunctionalGraph import *
from Components.DifferentiablePolynomial import *
from Models.ModelSinCos import *
from Models.ModelWeightedSum import *
from Models.ModelExponential import *

import numpy as np
import copy
from collections import defaultdict

def get_end_to_end_data(dg, gt_param):
    #ground truth end to end data from a graph
    dg = copy.deepcopy(dg)
    for node_idx in dg.nodes:
        dg.nodes[node_idx]["component"].set_params(gt_param[node_idx])
    X_local = np.random.uniform(0, 5, size=(100,3))
    y = []
    for x in X_local:
        y.append(dg.forward({1: x[0], 2: x[1], 7: x[2]}, "Blackbox6")) # need to infer entry and exit nodes from graph
    return torch.tensor(X_local), torch.tensor(y)
def show_system_loss_from_grad_descent(DG, ground_truth_param):
    X_end, y_end = get_end_to_end_data(DG, ground_truth_param)

    losses = defaultdict(list)
    num_itr = 200
    # gradient descent of individual
    for i in range(num_itr):
        for node_idx in DG.nodes:

            if "Blackbox" in str(node_idx):
                continue
            comp = DG.nodes[node_idx]["component"]
            comp.do_one_descent_on_local()
            losses[node_idx].append(comp.get_local_loss().detach().numpy())
        print(DG.get_system_loss())
        losses["system"].append(DG.get_system_loss(X_end, y_end))

    for l in losses:
        if l == 1:
            continue
        plt.plot(range(num_itr), losses[l], label=str(l)+" loss")
    plt.legend()
    plt.title("component and system loss with local gradient descent")
    plt.show()