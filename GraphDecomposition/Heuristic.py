import networkx as nx
import matplotlib.pyplot as plt

def plot(G):
    pos = nx.spectral_layout(G)
    nx.draw(G, pos=pos, with_labels = True)
    plt.show()

def get_backward_search_nodes(G):
    '''

    :param G: graph
    :return: gets all backward nodes from all nodes based on BFS
    '''
    all_simple_path = (nx.all_simple_paths(G, "BlackboxA", "exit"))
    backward_search = {}
    # for each forward search, do backward search
    for s in all_simple_path:
        G_temp = G.copy()
        G_temp.remove_node("BlackboxA")
        s.pop(0) # black box component
        for node in s:
            if node in backward_search:
                continue
            backward_search[node] = []
            backward_pass_from_forward_node = nx.bfs_edges(G_temp.reverse(), source=node)
            for x in backward_pass_from_forward_node:
                backward_search[node].append(x)

    backward_pass_from_backbox = nx.bfs_edges(G.reverse(), source="BlackboxA")
    backward_search["BlackboxA"]=[]
    for x in backward_pass_from_backbox:
        backward_search["BlackboxA"].append(x)
    return backward_search


def get_all_backward_decomposition(black_box_node_name, backward_search):
    backward_decompositions = []
    decomposition = [black_box_node_name]

    for edge in backward_search[black_box_node_name]:
        backward_decompositions.append(decomposition.copy())
        decomposition.append(edge[1])
    backward_decompositions.append(decomposition.copy())

    return backward_decompositions


def generate_forward_decompositions(G, simple_path_from_black_box_to_exit, backward_search):
    forward_decompositions = []
    decomposition = []
    explored_nodes = set()
    simple_path_from_black_box_to_exit = list(simple_path_from_black_box_to_exit)
    print("Number of path from black box to exit node: ", len(simple_path_from_black_box_to_exit))
    for s in simple_path_from_black_box_to_exit:
        G_temp = G.copy()
        G_temp.remove_node("BlackboxA")
        s.pop(0) # black box component
        for node in s:
            if node in explored_nodes:
                continue
            decomposition.append(node)
            forward_decompositions.append(decomposition.copy())
            for edge in backward_search[node]:
                if edge[1] in explored_nodes:
                    break
                decomposition.append(edge[1])
                forward_decompositions.append(decomposition.copy())
            explored_nodes.add(node)
    return forward_decompositions