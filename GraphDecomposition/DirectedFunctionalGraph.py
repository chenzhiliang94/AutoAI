import networkx as nx
import torch
import warnings
import torch.nn as nn

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

    def forward(self, sources:dict, sink):
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
                return component(x)
            assert "parents" in self.nodes[node]
            input = []
            for i, parent in enumerate(self.nodes[node]["parents"]):
                assert parent is not None, f"Input {i} for node {node} is required but not provided"
                input.append(backward_(parent))
            input = torch.tensor(input)
            return component(input)
        return backward_(sink)

    def get_system_loss_with_inputs(self, X, y):
        mse = nn.MSELoss()
        y_pred = []
        for x in X:
            y_pred.append(self.forward({1: x[0], 2: x[1], 7: x[2]}, "Blackbox6"))
        loss = mse(torch.tensor(y_pred), y)
        return loss

    def get_system_loss(self):
        return self.get_system_loss_with_inputs(self.system_x, self.system_y)

    def get_local_losses(self):
        losses = []
        for n in self.nodes:
            if "Blackbox" in str(n):
                continue
            losses.append(self.nodes[n]["components"].get_local_loss())
        return losses

    def get_num_components(self):
        component = [x for x in self.nodes if "Blackbox" not in str(self.nodes[x])]
        return len(component)
    def get_all_params(self):
        dict_ = {}
        param = []
        for n in self.nodes:
            if "Blackbox" in str(n):
                continue
            dict_[n] = self.nodes[n]["component"].get_params()
            param += list(self.nodes[n]["component"].get_params())
        return dict_, param

    def assign_params(self, params : dict):
        for n in self.nodes:
            if "Blackbox" in str(n):
                continue
            self.nodes[n]["component"].set_params(params[n])

    def assign_params(self, params):
        for n in self.nodes:
            if "Blackbox" in str(n):
                continue
            num_param_to_assign = len(self.nodes[n]["component"].get_params())
            self.nodes[n]["component"].set_params(params[:num_param_to_assign])
            params = params[num_param_to_assign:]