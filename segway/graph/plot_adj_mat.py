
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import moral
from networkx.utils import reverse_cuthill_mckee_ordering
import copy


def remove_exclusion_list(synapse_graph, full_list):
    """Exclude specified nodes from Adj in config file."""
    pre_exclusion_list = synapse_graph.get_presynapse_exclusion_list()
    post_exclusion_list = synapse_graph.get_postsynapse_exclusion_list()
    pre_list = copy.deepcopy(full_list)
    post_list = copy.deepcopy(full_list)
    print(pre_list)
    print(post_list)
    for n in pre_exclusion_list:
        print(n)
        pre_list.remove(n)
    for n in post_exclusion_list:
        print(n)
        post_list.remove(n)

    return pre_list, post_list


def plot_adj_mat(synapse_graph, configs, sort_type=None):  # A, configs, g):
    """
    Plot Adj matrix according to the configs.

    - configs is a dictionary with the following keys:
    ['full', 'some', 'pre', 'post', 'threshold_value'(int/float)];
    - the output paths are the values of the dict;
    configs values will include the list of neuorns of interest.
    - sort_type sorts the matrix according to the specified type;
    if not None, it can be 'labels', 'patterns', 'sort'. If the type is patterns,
    the graph will be converted into an undirected graph to find blocks.
    Note: If the threshold is present, all the plots will be done considering
    that threshold.
    """
    A = synapse_graph.get_matrix()
    graph = synapse_graph.get_graph()

    if 'weights_threshold_min' in configs:
        A[A < configs['weights_threshold_min']] = 0
    if 'weights_threshold_max' in configs:
        A[A > configs['weights_threshold_max']] = configs['weights_threshold_max']

    full_list = list(graph.nodes())
    fig = plt.figure(figsize=(16, 15))
    ax = fig.add_subplot(111)

    if configs['analysis_type'] == 'adj_plot_all':
        if sort_type == 'patterns':
            ug = moral.moral_graph(graph)
            rcm = list(reverse_cuthill_mckee_ordering(ug))
            full_list = rcm
            A = nx.adjacency_matrix(graph, nodelist=rcm).todense()
            if 'weights_threshold_min' in configs:
                A[A < configs['weights_threshold_min']] = 0
            if 'weights_threshold_max' in configs:
                A[A > configs['weights_threshold_max']] = configs['weights_threshold_max']

        pre_list, post_list = remove_exclusion_list(synapse_graph, full_list)
        if sort_type == 'labels':
            pre_list = sorted(pre_list)
            post_list = sorted(post_list)

        mat = A[
            [full_list.index(i) for i in pre_list], :
        ]
        mat = mat[
            :, [full_list.index(i) for i in post_list]
        ]

        if sort_type == 'sort':
            mat = np.sort(mat)

        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(post_list, rotation=90)
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(pre_list)

        ax.grid(True, alpha=0.2)

    elif configs['analysis_type'] == 'adj_plot_pre':
        mat = A[:, [full_list.index(i) for i in configs['list']]]
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(configs['list'], rotation=90)
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(full_list)

        # save output file for synapses proofreading
        small_list = configs['list']
        synapse_graph.debug_spec_edges(pre_list=small_list)

    elif configs['analysis_type'] == 'adj_plot_post':
        mat = A[[full_list.index(i) for i in configs['list']], :]
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(full_list, rotation=90)
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(configs['list'])

        # save output file for synapses proofreading
        small_list = configs['list']
        synapse_graph.debug_spec_edges(post_list=small_list)

    elif configs['analysis_type'] == 'adj_plot_some':
        mat = nx.to_numpy_matrix(graph, nodelist=configs['list'])
        ax.set_xticks(np.arange(len(configs['list'])))
        ax.set_xticklabels(configs['list'], rotation=90)
        ax.set_yticks(np.arange(len(configs['list'])))
        ax.set_yticklabels(configs['list'])

        # save output file for synapses proofreading
        small_list = configs['list']
        synapse_graph.debug_spec_edges(pre_list=small_list, post_list=small_list)

    else:
        print("### Info: analysis_type specified not implemented! Exiting...")
        exit()

    i = ax.imshow(mat)
    plt.colorbar(i, ax=ax)
    fig.savefig(synapse_graph.directory + '/' + configs['output_plot'])
