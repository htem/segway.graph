import numpy as np
from graph_tool.all import *


""" G is the graph in graph tool and gnx in networkx"""

def create_edges_graph_gt(G, edge_list_ids):
    """Create graph in GT through edge_list_ids."""
    if len(edge_list_ids[0]) == 2:
        G.add_edges_from(edge_list)
        print("### Info : Edges created!")

    elif len(edge_list_ids[0]) == 3:
        # the weights are the 3rd column
        weight = G.new_edge_property('double')
        G.add_edge_list(edge_list_ids)
        w = []
        for el in edge_list_ids:
            w.append(el[2])
        weight.a = w
        G.ep['weight'] = weight

        print("### Info : Edges and weights created!")

    else:
        print("ERROR: edge_list has to be in (pre, post) format: third entry should be the\
               corresponding weight!")

    return G

def convert_el_to_ids(edge_list, neurons_to_ids):
    """Input neurons_to_ids is n_dic."""
    edge_list_ids = []

    if len(edge_list[0]) == 2:
        for e in edge_list:
            if e[0] in neurons_to_ids.keys() and e[1] in neurons_to_ids.keys():
                edge_list_ids.append((neurons_to_ids[e[0]], neurons_to_ids[e[1]]))

    elif len(edge_list[0]) == 3:
        for e in edge_list:
            if e[0] in neurons_to_ids.keys() and e[1] in neurons_to_ids.keys():
                edge_list_ids.append((neurons_to_ids[e[0]], neurons_to_ids[e[1]], e[2]))

    else:
        print("ERROR: edge_list has to be in (pre, post) format: third entry should be the\
               corresponding weight!")

    return edge_list_ids

def create_nodes_and_attr_gt(gnx):
    """Create the nodes of the graph with useful attribute/properties in graph-tool."""
    n_dic = create_neurons_to_ids(gnx)
    id_nodes = np.array(list(n_dic.values()))

    G = Graph(directed=True)
    G.add_vertex(len(id_nodes))

    cell_type = G.new_vertex_property("string")
    G.vertex_properties['cell_type'] = cell_type
    node_name = G.new_vertex_property("string")
    G.vertex_properties['node_name'] = node_name
    pos_x = G.new_vertex_property("float")
    G.vertex_properties['pos_x'] = pos_x
    pos_y = G.new_vertex_property("float")
    G.vertex_properties['pos_y'] = pos_y
    pos_z = G.new_vertex_property("float")
    G.vertex_properties['pos_z'] = pos_z
    pos = G.new_vertex_property("vector<float>")
    G.vertex_properties['pos'] = pos

    # networkx attributes
    nx_att_celltype = dict(gnx.nodes(data='cell_type'))
    nx_att_x = dict(gnx.nodes(data='x'))
    nx_att_y = dict(gnx.nodes(data='y'))
    nx_att_z = dict(gnx.nodes(data='z'))

    for i, nid in enumerate(gnx.nodes()):
        cell_type[i] = nx_att_celltype[nid]
        pos_x[i] = nx_att_x[nid]
        pos_y[i] = nx_att_y[nid]
        pos_z[i] = nx_att_z[nid]
        pos[i] = np.array([nx_att_x[nid], nx_att_y[nid]])
        node_name[i] = nid

    return G, n_dic

def create_neurons_to_ids(gnx):
    """Create ids mapping the neurons which will be the nodes."""
    n_dic = dict()
    for i, n in enumerate(gnx.nodes()):
        n_dic[n] = i

    return n_dic

def create_ids_to_neurons(n_dic):
    ids_to_neurons = {}
    for k,v in n_dic.items():
        ids_to_neurons[v] = k

    return ids_to_neurons
