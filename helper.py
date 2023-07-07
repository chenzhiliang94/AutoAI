from GraphDecomposition.Heuristic import *
from GraphDecomposition.DirectedFunctionalGraph import *
from Components.DifferentiablePolynomial import *
from Models.ModelSinCos import *
from Models.ModelWeightedSum import *
from Models.ModelExponential import *
from Models.ModelConstant import *
from GraphDecomposition.MutualInformation import mutual_information_nodes_samples

import numpy as np
import copy
from collections import defaultdict
from sklearn.feature_selection import mutual_info_regression

def get_end_to_end_data(dg, gt_param):
    #ground truth end to end data from a graph
    dg = copy.deepcopy(dg)
    for node_idx in dg.nodes:
        if node_idx in gt_param:
            dg.nodes[node_idx]["component"].set_params(gt_param[node_idx])
    
    y = []
    exit = dg.get_exit()
    entry = dg.get_entry()
    X_local = np.random.uniform(0, 5, size=(100,len(entry)))

    for x in X_local:
        assert len(x) == len(entry)
        input_dict = {}
        for node_idx, input in zip(entry, x):
            input_dict[node_idx] = input
        y.append(dg.forward(input_dict, exit, perturbed_black_box=False)) # when generating data, no perturbation
    return torch.tensor(X_local), torch.tensor(y)

'''
Given a 1) DG, 2) a set of sub-system 3)ground truth param,
Generate a new DG with end to end data inside
'''
def generate_sub_system(sub_system : set, DG : DirectedFunctionalGraph, ground_truth_param : dict) -> DirectedFunctionalGraph:
    sub_system_extended = DG.generate_sub_system(sub_system)
    graph_temp = copy.deepcopy(DG)
    graph_temp.retain_nodes(sub_system_extended)
    relabel_nodes = {}
    for n in graph_temp.nodes:
        if n not in sub_system:
            graph_temp.nodes[n]["component"] = ModelConstant()
            relabel_nodes[n] = str(n) + "Dummy"
            graph_temp = nx.relabel_nodes(graph_temp, relabel_nodes)
    
    # reassign parents to proper name
    for n in graph_temp.nodes:
        if "parents" in graph_temp.nodes[n]:
            new_parents = []
            for parent in graph_temp.nodes[n]["parents"]:
                if parent in relabel_nodes:
                    new_parents.append(relabel_nodes[parent])
                else:
                    new_parents.append(parent)
            graph_temp.nodes[n]["parents"] = new_parents
            

    X,y = get_end_to_end_data(graph_temp,ground_truth_param)
    graph_temp.system_x = X
    graph_temp.system_y = y
    return graph_temp

'''
Do gradient descent on every non-black box component and plot the losses, along with system loss
'''
def show_system_loss_from_grad_descent(DG, itr=500, plot=False):
    losses = defaultdict(list)
    num_itr = itr
    # gradient descent of individual
    for i in range(num_itr):
        last_loss = []
        for node_idx in DG.nodes:

            if ("Dummy" in str(node_idx)) or ("Blackbox" in str(node_idx)):
                continue
            comp = DG.nodes[node_idx]["component"]
            comp.do_one_descent_on_local()
            local_loss = comp.get_local_loss().detach()
            losses[node_idx].append(local_loss)
            last_loss.append(local_loss)
        losses["system"].append(DG.get_system_loss())
    
    if plot:
        for l in losses:
            plt.plot(range(num_itr), losses[l], label=str(l)+" loss")
        plt.legend()
        plt.ylim(0,3)
        plt.title("component and system loss with local gradient descent")
        plt.show()
    return last_loss

'''
Find M.I of each component loss w.r.t system loss
via random sampling (uniform w.r.t gradient descent loss)
'''
def get_mi(dg : DirectedFunctionalGraph):
    param_original = copy.deepcopy(dg.get_all_params()[1])
    samples = mutual_information_nodes_samples(dg)
    dg.assign_params(param_original)

    all_local_losses = []
    all_system_losses = []
    for x in range(100):
        l = np.array(dg.get_local_losses()).flatten()
        all_local_losses.append(l)
        L = dg.get_system_loss()
        all_system_losses.append(L)
        for node_name in samples:
            local_node_loss = samples[node_name].keys()
            sampled_local_loss = sample(list(local_node_loss), 1)[0]
            sampled_param = samples[node_name][sampled_local_loss]
            dg.assign_param_to_node(node_name, sampled_param)
        

    all_local_losses = np.stack(all_local_losses)
    mi = mutual_info_regression(all_local_losses, all_system_losses, n_neighbors=10)
    return mi

'''
Compute M.I of each component and assign to each node of DG
'''
def assign_mi(DG : DirectedFunctionalGraph):
    DG.random_initialize_param()
    mi = get_mi(DG)
    mi_dict = {k:v for k,v in zip(list(DG.get_components().keys()), mi)}
    DG.assign_mutual_information_to_node(mi_dict)
    print("MI: ", mi)
    return DG