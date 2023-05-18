import networkx as nx
import torch
import warnings

class DirectedFunctionalGraph(nx.DiGraph):
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
        if required_inputs > 1:
            assert len(u_of_edge) == required_inputs, f"node {v_of_edge} require {required_inputs} parents, only {len(u_of_edge)} supplied"
            for u in u_of_edge:
                if not u is None:
                    super().add_edge(u, v_of_edge, **attr)
        else:
            super().add_edge(u_of_edge[0], v_of_edge, **attr)
        self.nodes[v_of_edge]["parents"] = u_of_edge

    def forward(self, sources:dict, sink):
        def backward_(node):
            component = self.nodes[node]["component"]
            if node in sources:
                x = sources[node]
                if not torch.is_tensor(x):
                    x = torch.tensor(x)
                return component(x)
            assert "parents" in self.nodes[node]
            input = [backward_(parent) for parent in self.nodes[node]["parents"]]
            input = torch.tensor(input)
            return component(input)
        return backward_(sink)
