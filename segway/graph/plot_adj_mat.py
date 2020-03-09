import os
import numpy as np
import matplotlib.pyplot as plt
# import networkx as nx
import copy
import re

import networkx as nx
from networkx.algorithms import moral
from networkx.utils import reverse_cuthill_mckee_ordering


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def remove_exclusion_list(synapse_graph, pre, post):
    pre_exclusion_list = synapse_graph.get_presynapse_exclusion_list()
    post_exclusion_list = synapse_graph.get_postsynapse_exclusion_list()
    pre_list = copy.deepcopy(pre)
    post_list = copy.deepcopy(post)
    for n in pre_exclusion_list:
        if n in pre_list:
            pre_list.remove(n)
    for n in post_exclusion_list:
        if n in post_list:
            post_list.remove(n)

    return pre_list, post_list


class PlotConfig():

    def __init__(
            self, configs, full_list=None, dir=None, synapse_graph=None,
            plot_type='default'):

        self.threshold_min = configs.get('weights_threshold_min', None)
        self.threshold_max = configs.get('weights_threshold_max', None)
        self.colorbar = configs.get('colorbar', False)
        # self.plot_type = configs.get('plot_type', None)
        self.save_edges_to_csv = configs.get('save_edges_to_csv', True)

        full_list = configs.get('full_list', configs.get('list', full_list))
        assert full_list is not None
        self.pre_list = configs.get('pre_list', full_list)
        self.post_list = configs.get('post_list', full_list)

        self.fname = configs.get('fname', plot_type)

        self.sort = configs.get('sort', 'labels')

        self.synapse_graph = synapse_graph

        self.also_plot_transposed = configs.get('also_plot_transposed', False)
        self.plot_width = configs.get('plot_width', 14.5)
        self.plot_height = configs.get('plot_height', 14.5)

    def get_output_fname(self, arg):
        if self.threshold_min:
            arg = arg + '_min_' + str(self.threshold_min)
        return self.synapse_graph.get_output_fname(arg)

    def get_output_csv_fname(self, arg):
        if self.threshold_min:
            arg = arg + '_min_' + str(self.threshold_min)
        return self.synapse_graph.get_output_csv_fname(arg)


def plot_adj_mat(synapse_graph, configs):
    """
    Plot Adj matrix according to the configs.

    - configs is a dictionary with the following keys:
    ['full', 'some', 'pre', 'post', 'threshold_value'(int/float)];
    - the output paths are the values of the dict;
    configs values will include the list of neuorns of interest.

    If the threshold is present, all the plots will be done considering
    that threshold.
    """
    graph = synapse_graph.get_graph()
    full_list = list(graph.nodes())
    plot_config = PlotConfig(
        configs, full_list=full_list, dir=synapse_graph.output_dir, synapse_graph=synapse_graph,
        plot_type='adj_plot')
    A = copy.deepcopy(synapse_graph.get_matrix())  # need to preserve A for subsequent plots

    if plot_config.threshold_min is not None or plot_config.threshold_max is not None:
        if plot_config.threshold_min is not None:
            A[A < plot_config.threshold_min] = 0
        if plot_config.threshold_max is not None:
            A[A > plot_config.threshold_max] = plot_config.threshold_max

    pre_list = synapse_graph.expand_list(plot_config.pre_list)
    print("plot_config.post_list:", plot_config.post_list)
    post_list = synapse_graph.expand_list(plot_config.post_list)
    print("post_list:", post_list)

    pre_list, post_list = remove_exclusion_list(synapse_graph, pre_list, post_list)

    pre_list_renamed = synapse_graph.rename_list(pre_list)
    post_list_renamed = synapse_graph.rename_list(post_list)
    full_list_renamed = synapse_graph.rename_list(full_list)

    # print(pre_list_renamed)
    print("post_list_renamed:", post_list_renamed)
    # print(full_list_renamed)

    if plot_config.sort is not None:
        if plot_config.sort == 'patterns':
            ug = moral.moral_graph(graph)
            rcm = list(reverse_cuthill_mckee_ordering(ug))
            pre_list_renamed = [n for n in rcm if n in pre_list_renamed]
            post_list_renamed = [n for n in rcm if n in post_list_renamed]
            # pre_list_renamed = full_list
        elif plot_config.sort == 'labels':
            pre_list_renamed = sorted(pre_list_renamed, key=natural_keys)
            post_list_renamed = sorted(post_list_renamed, key=natural_keys)
        elif plot_config.sort == 'sort':
            assert False, "This option is not properly implemented"
            # TODO: need to also sort labels along with mat
            mat = np.sort(mat)
        else:
            # prelist and postlist should have the same order as the full list
            pre_list_renamed = [n for n in full_list_renamed if n in pre_list_renamed]
            post_list_renamed = [n for n in full_list_renamed if n in post_list_renamed]

    mat = A[
        [full_list_renamed.index(name) for name in pre_list_renamed], :
    ]
    mat = mat[
        :, [full_list_renamed.index(name) for name in post_list_renamed]
    ]

    _plot_adj_mat(mat, pre_list_renamed, post_list_renamed, plot_config, synapse_graph, transposed=False, colorbar=plot_config.colorbar)
    if plot_config.also_plot_transposed:
        _plot_adj_mat(mat, pre_list_renamed, post_list_renamed, plot_config, synapse_graph, transposed=True, colorbar=plot_config.colorbar)


def _plot_adj_mat(
        mat, pre_list, post_list, plot_config,
        synapse_graph, transposed=False, colorbar=False):

    # fig = plt.figure(figsize=(16, 15))
    if not transposed:
        fig = plt.figure(figsize=(plot_config.plot_width, plot_config.plot_height))
    else:
        fig = plt.figure(figsize=(plot_config.plot_height, plot_config.plot_width))
    ax = fig.add_subplot(111)

    if transposed:
        post_list0 = post_list
        post_list = pre_list
        pre_list = post_list0
        mat = mat.transpose()

    ax.set_xticks(range(mat.shape[1]), minor=True)
    ax.set_xticklabels(post_list, rotation=90, minor=True)
    ax.set_yticks(range(mat.shape[0]), minor=True)
    ax.set_yticklabels(pre_list, minor=True)

    ax.set_yticks(range(0, mat.shape[0], 5), minor=False)
    ax.set_yticklabels([], minor=False)
    ax.set_xticks(range(0, mat.shape[1], 5), minor=False)
    ax.set_xticklabels([], minor=False)
    ax.grid(True, which='major', alpha=0.5)
    ax.grid(True, which='minor', alpha=0.2)
    ax.tick_params(axis='both', which='both', labelsize=8)
    i = ax.imshow(mat)
    if colorbar:
        plt.colorbar(i, ax=ax)
    plt.tight_layout()
    # fig.savefig(synapse_graph.output_dir + '/' + configs['output_plot'])

    fname = plot_config.fname
    if transposed:
        fname = fname + '_transposed'
    fig.savefig(plot_config.get_output_fname(fname))

    if plot_config.save_edges_to_csv:
        synapse_graph.save_edges_to_csv(pre_list, post_list, plot_config.get_output_csv_fname(fname))
