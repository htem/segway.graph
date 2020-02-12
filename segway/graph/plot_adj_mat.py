
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_adj_mat(synapse_graph, configs):  # A, configs, g):
    """
    Plot Adj matrix according to the configs.

    - configs is a dictionary with the following keys:
    ['full', 'some', 'pre', 'post', 'threshold_value'(int/float)];
    - the output paths are the values of the dict;
    configs values will include the list of neuorns of interest.

    If the threshold is present, all the plots will be done considering
    that threshold.
    """
    A = synapse_graph.get_matrix()
    graph = synapse_graph.get_graph()

    if configs['adj_plot_thresh'] == 1:
        A[A <= configs['weights_threshold']] = 0

    full_list = list(graph.nodes())
    fig = plt.figure(figsize=(16, 15))
    ax = fig.add_subplot(111)

    if configs['analysis_type'] == 'adj_plot_all':
        mat = A
        ax.set_xticks(np.arange(len(mat)))
        ax.set_xticklabels(full_list, rotation=75)
        ax.set_yticks(np.arange(len(mat)))
        ax.set_yticklabels(full_list)
    elif configs['analysis_type'] == 'adj_plot_pre':
        mat = A[:, [full_list.index(i) for i in configs['list']]]
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(configs['list'], rotation=75)
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(full_list)

        # save output file for synapses proofreading
        small_list = configs['list']
        synapse_graph.debug_spec_edges(pre_list=small_list)

    elif configs['analysis_type'] == 'adj_plot_post':
        mat = A[[full_list.index(i) for i in configs['list']], :]
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(full_list, rotation=75)
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(configs['list'])

        # save output file for synapses proofreading
        small_list = configs['list']
        synapse_graph.debug_spec_edges(post_list=small_list)

    elif configs['analysis_type'] == 'adj_plot_some':
        mat = nx.to_numpy_matrix(graph, nodelist=configs['list'])
        ax.set_xticks(np.arange(len(configs['list'])))
        ax.set_xticklabels(configs['list'], rotation=75)
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