import itertools
import copy

import networkx as nx
import torch
import warnings
import torch.nn as nn
import numpy as np
from random import sample
from collections import OrderedDict

from Models.ModelConstant import ModelConstant
class DirectedFunctionalGraph(nx.DiGraph):
    system_x = None
    system_y = None
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """Add an edges from each u to v.
        Parameters
        ----------
        u_of_edge : Ordered list of input nodes

        v_of_edge : node
        """
        if not (isinstance(u_of_edge, list) or isinstance(u_of_edge, tuple)):
            u_of_edge = [u_of_edge]
        if "parents" in self.nodes[v_of_edge]:
            warnings.warn(f'Parents of {v_of_edge} previously defined as {self.nodes[v_of_edge]["parents"]}, attempting to overwrite with {u_of_edge}')
            for edge in self.nodes[v_of_edge]["parents"]:
                self.remove_edge(edge, v_of_edge)
            self.nodes[v_of_edge].pop("parents")
        required_inputs = self.nodes[v_of_edge]["component"].inputs
        assert len(u_of_edge) == required_inputs, f"node {v_of_edge} require {required_inputs} parents, but {len(u_of_edge)} supplied"
        if required_inputs > 1:
            for u in u_of_edge:
                if u is not None:
                    print(f"adding edge from {u} to {v_of_edge}")
                    super().add_edge(u, v_of_edge, **attr)
        else:
            super().add_edge(u_of_edge[0], v_of_edge, **attr)
        self.nodes[v_of_edge]["parents"] = u_of_edge

    def forward(self, sources:dict, sink, perturbed_black_box=False):
        def backward_(node):
            component = self.nodes[node]["component"]
            if node in sources:
                
                x = sources[node]
                if component.inputs > 1:
                    for i in range(component.inputs):
                        if x[i] is None:    # Input not provided, query upwards
                            assert "parents" in self.nodes[node], f"Parents for node {node} required but not defined"
                            assert self.nodes[node]["parents"][i] is not None, f"Parent {i} for node {node} required but not defined"
                            x[i] = backward_(self.nodes[node]["parents"][i])
                if not torch.is_tensor(x):
                    x = torch.tensor(x)
                if isinstance(component, ModelConstant): 
                    return component(x,noisy=False)
                if "Blackbox" in str(node):
                    if perturbed_black_box:
                        return component(x,noisy=True, noise_mean=0.1)
                    return component(x,noisy=True, noise_mean=0.0)
                return component(x,noisy=False)
            
            assert "parents" in self.nodes[node]
            input = []
            for i, parent in enumerate(self.nodes[node]["parents"]):
                assert parent is not None, f"Input {i} for node {node} is required but not provided"
                input.append(backward_(parent))
            input = torch.tensor(input)
            component = self.nodes[node]["component"]
            if isinstance(node, ModelConstant): 
                    return component(input,noisy=False)
            if "Blackbox" in str(node):
                if perturbed_black_box:
                    return component(input,noisy=True, noise_mean=0.1)
                return component(input,noisy=True, noise_mean=0.0)
            return component(input, noisy=True)
        return backward_(sink)
    
    def generate_sub_system(self, node_set : set):
        extended_node_set = set()
        for n in node_set:
            extended_node_set.add(n)
            extended_node_set = extended_node_set | set(list(self.predecessors(n)))
        return extended_node_set
        
    def retain_nodes(self, node_set : set):
        all_nodes = copy.deepcopy(self.nodes)
        for n in all_nodes:
            if n not in node_set:
                self.remove_node(n)
                for node in self.nodes:
                    if "parents" not in self.nodes[node]:
                        continue
                    parents : list = self.nodes[node]["parents"]
                    if n in parents:
                        parents.remove(n)
                    self.nodes[node]["parents"] = parents

    def get_exit(self):
        exit = [n for n,d in self.out_degree() if d==0][0]
        return exit

    def get_entry(self):
        entry = [n for n,d in self.in_degree() if d==0]
        return entry
        
    def get_system_loss_with_inputs(self, X, y):
        mse = nn.MSELoss()
        y_pred = []
        exit = self.get_exit()
        entry = self.get_entry()
        for x in X:
            assert len(x) == len(entry)
            input_dict = {}
            for node_idx, input in zip(entry, x):
                input_dict[node_idx] = input
            y_pred.append(self.forward(input_dict, exit, perturbed_black_box=True))
        loss = mse(torch.tensor(y_pred), y)
        return loss

    def get_system_loss(self):
        return self.get_system_loss_with_inputs(self.system_x, self.system_y)

    def get_local_losses(self):
        losses = []
        for n in self.nodes:
            if "Blackbox" in str(n) or "Dummy" in str(n):
                continue
            losses.append(self.nodes[n]["component"].get_local_loss())
        return torch.DoubleTensor([losses])

    def get_components(self):
        components = OrderedDict((x, self.nodes[x]) for x in self.nodes if "Dummy" not in str(x) and "Blackbox" not in str(x))
        return components
    def get_all_params(self):
        dict_ = {}
        param = []
        for n in self.nodes:
            if "Blackbox" in str(n) or "Dummy" in str(n):
                continue
            dict_[n] = self.nodes[n]["component"].get_params()
            param += list(self.nodes[n]["component"].get_params())
        return dict_, param

    def assign_params(self, params : dict):
        for n in self.nodes:
            if "Blackbox" in str(n) or "Dummy" in str(n):
                continue
            self.nodes[n]["component"].set_params(params[n])

    def assign_param_to_node(self, node : str, param : list):
        self.nodes[node]["component"].set_params(param)

    def assign_params(self, params : list):
        for n in self.nodes:
            if "Blackbox" in str(n) or "Dummy" in str(n):
                continue
            num_param_to_assign = len(self.nodes[n]["component"].get_params())
            self.nodes[n]["component"].set_params(params[:num_param_to_assign])
            params = params[num_param_to_assign:]

    # look for parameters which yield the input losses
    def reverse_local_loss_lookup(self, losses, method):
        components = self.get_components().values()
        assert len(components) == len(losses), "loss input size should be equals to number of components!"

        norm_difference = torch.subtract(self.get_local_losses(), torch.tensor(losses))
        #print("loss difference norm before: ", torch.norm(norm_difference))
        #print("target losses: ", losses)
        #print("current losses: ", self.get_local_losses())

        # do gradient ascent/descent until loss is reached from current parameter configuration
        if method == "naive_climb":
            for loss_comp in zip(components, losses):
                component = loss_comp[0]["component"]
                loss_target = loss_comp[1]

                itr = 0
                while itr < 1000 and abs(component.get_local_loss() - loss_target) / loss_target > 1e-02: # % threshold
                    curr_loss = component.get_local_loss()
                    if curr_loss > loss_target:
                        component.do_one_descent_on_local()
                    else:
                        component.do_one_ascent_on_local() # can replace this simply with another custom loss function
                    next_loss = component.get_local_loss()
                    itr+=1
                    if abs(next_loss - curr_loss) < 1e-02:
                        break
            best_param = self.get_all_params()[1]

        # initialise multiple parameter initialization and search for the best
        if method == "multi_search":
            print("loss to look for: ", losses)
            params = []
            num_starting_points = 30
            sample_size = 5
            final_sample = 20
            for loss_comp in zip(components, losses):
                param_candidates = []
                for n in range(num_starting_points):
                    component = loss_comp[0]["component"]
                    loss_target = loss_comp[1]
                    component.random_initialise_params()
                    itr = 0
                    while itr < 500 and abs(component.get_local_loss() - loss_target) / loss_target > 1e-01: # % threshold
                        curr_loss = component.get_local_loss()
                        if curr_loss > loss_target:
                            component.do_one_descent_on_local()
                        else:
                            component.do_one_ascent_on_local() # can replace this simply with another custom loss function
                        next_loss = component.get_local_loss()
                        itr+=1
                    if abs(component.get_local_loss() - loss_target) / loss_target > 1e-01:
                        continue
                    if list(component.get_params()) not in param_candidates:
                        param_candidates.append(list(component.get_params()))
                assert len(param_candidates) > 0
                while len(param_candidates) < sample_size:
                    param_candidates = param_candidates + param_candidates
                params.append(sample(param_candidates,sample_size)) # sample
            all_idx_combination = len(params) * [[x for x in range(sample_size)]]
            all_cartesian_idx = list(itertools.product(*all_idx_combination))
            a = len(list(all_cartesian_idx))
            b = final_sample
            min_to_sample = min(a, b)
            print(a)
            print(b)
            print(min_to_sample)
            print("number to sample: ", min_to_sample)
            all_cartesian_idx = sample(all_cartesian_idx, min_to_sample) # sample again
            # param_configuration_to_check = sample(list(all_cartesian_idx), 20)
            # print(param_configuration_to_check)
            print(all_cartesian_idx)
            
            best_system_loss = 1e50
            best_param = None
            count = 0
            for idx in all_cartesian_idx:
                count+=1
                candidate_param = []
                for i, cartesian_idx in enumerate(list(idx)):
                    candidate_param += params[i][cartesian_idx]
                self.assign_params(candidate_param)
                candidate_system_loss = self.get_system_loss()
                if candidate_system_loss < best_system_loss:
                    best_system_loss = candidate_system_loss
                    best_param = candidate_param
            print("number of combination of params with matching loss: ", count)
        
        if method == "block_minimization":
            params = []
            num_starting_points = 2
            for loss_comp in zip(components, losses):
                param_candidates = []
                for n in range(num_starting_points):
                    component = loss_comp[0]["component"]
                    loss_target = loss_comp[1]
                    component.random_initialise_params()
                    itr = 0
                    while itr < 500 and abs(component.get_local_loss() - loss_target) / loss_target > 1e-02: # % threshold
                        curr_loss = component.get_local_loss()
                        if curr_loss > loss_target:
                            component.do_one_descent_on_local()
                        else:
                            component.do_one_ascent_on_local() # can replace this simply with another custom loss function
                        next_loss = component.get_local_loss()
                        itr+=1
                        if abs(next_loss - curr_loss) < 1e-05:
                            break
                    if list(component.get_params()) not in param_candidates:
                        param_candidates.append(list(component.get_params()))
                assert len(param_candidates) > 0
                params.append(param_candidates)

            # might need to assign initial params
            
            # perform block minimization
            node_names = [x for x in self.nodes if not ("Blackbox" in str(x) or "Dummy" in str(x))]
            for i in range(20): # iterations
                for node_name, param_candidates in zip(node_names, params):
                    # param_candidate is a list of params, iterate and assign best
                    best_system_loss = 1000
                    for param_candidate in param_candidates:
                        self.assign_param_to_node(node_name, param_candidate)
                        curr_loss = self.get_system_loss()
                        if curr_loss < best_system_loss:
                            best_system_loss = curr_loss
            best_param = self.get_all_params()[1]
            print(best_param)
                        
        self.assign_params(best_param)
        norm_difference = torch.subtract(torch.tensor(self.get_local_losses()), torch.tensor(losses))
        #print("loss difference norm after: ", torch.norm(norm_difference))

    def fit_locally_partial(self, itr=50):
        for n in self.nodes:
            if "Blackbox" in str(n) or "Dummy" in str(n):
                continue
            for i in range(itr):
                self.nodes[n]["component"].do_one_descent_on_local()
    
    def random_initialize_param(self):
        param = self.get_all_params()[1]
        param = [np.random.uniform(-1,1) for x in param]
        self.assign_params(param)
    
    def assign_mutual_information_to_node(self, mi : dict):
        for node in mi:
            self.nodes[node]["mi"] = mi[node]
    
    def debug_loss(self):
        for x in range(10):
            self.fit_locally_partial(1)
            print("local losses: ", self.get_local_losses())
            print("system loss: ", self.get_system_loss())
            print("\n")