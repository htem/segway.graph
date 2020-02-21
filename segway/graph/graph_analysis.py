import numpy as np
import math
import networkx as nx
from graph_tool.all import *
import igraph as ig
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import itertools
import scipy
import matplotlib as mpl
from matplotlib import cm
import graph_tool_functions as gtf
from segway.graph.synapse_graph import SynapseGraph
import logging
from jsmin import jsmin
from io import StringIO


logger = logging.getLogger(__name__)

class GraphAnalysis():
    """GraphAnalysis """
    def __init__(self, config_file):
        super(GraphAnalysis, self).__init__()
        self.config_file = config_file

        self.__initialize_configs()
        self.__read_configs()

        os.makedirs(self.output_dir, exist_ok=True)

        self._convert_nx_to_gt()

    def __initialize_configs(self):
        """Initialize default values."""
        # create output directory with same name of config file
        output_dir, config_name = os.path.split(self.config_file)
        if output_dir == '':
            output_dir = '.'
        config_name = config_name.split('.')[0]
        output_dir = os.path.join(output_dir, config_name)

        self.output_dir = output_dir
        self.graphtool_plot = []
        self.networkx_plot = []
        self.analysis_plots = []
        self.count_motifs = []
        self.identify_triangles = []

    def __read_configs(self):
        """Recursively read configs from given JSON file."""
        logger.info("Graph analysis specified in config %s" % self.config_file)


        with open(self.config_file) as js_file:
            minified = jsmin(js_file.read())
            params = json.load(StringIO(minified))
            # params = json.load(f)

        assert 'graph_config_file' in params
        # read existing graph networkx
        neuronGraph = SynapseGraph(params['graph_config_file'])
        logger.info("Parsing graph from config %s" % params['graph_config_file'])
        self.gnx = neuronGraph.get_graph()
        print("### Info : read networkx graph ", self.gnx)
        self.neuron_list = neuronGraph.get_neurons_list()
        self.edge_list = self.extract_edge_list()
        self.edge_list_df = neuronGraph.edge_list_df
        self.A = neuronGraph.get_matrix()
        print(self.A.shape)
        for key in params:
            if hasattr(self, key) and isinstance(getattr(self, key), list):
                # extend if parameter is a list
                assert isinstance(params[key], list)
                # setattr(self, key, getattr(self, key).extend(params[key]))
                getattr(self, key).extend(params[key])
            else:
                # initialize or overwrite value
                setattr(self, key, params[key])

    def extract_edge_list(self):
        """Extract edge list from networkx."""

        edge_list = []
        for line in nx.generate_edgelist(self.gnx, delimiter=',', data=True):
            line = line.split(',')
            p = line[2].replace("{'weight': ", "")
            pp = p.replace('}', '')
            line[2] = float(pp)
            edge_list.append(line)

        return edge_list

    def _convert_nx_to_gt(self):
        self.gt, self.neurons_to_ids = gtf.create_nodes_and_attr_gt(self.gnx)

        print("### Number of nodes in graph tool: ", self.gt.num_vertices())
        self.edge_list_ids = gtf.convert_el_to_ids(self.edge_list, self.neurons_to_ids)
        self.gt = gtf.create_edges_graph_gt(self.gt, self.edge_list_ids)
        print("### Number of edges in graph tool: ", self.gt.num_edges())

    def plot_graphs(self):
        """Plot graphs in networkx and graph-tool according to the specified configs"""
        if len(self.graphtool_plot) > 0 :
            print("Plotting graph in graph_tool ...")

        for plot in self.graphtool_plot:
            pos_xy = plot.get('pos', False)
            deg = plot.get('degree', False)
            fname = plot.get('fname', 'graph_gt')

            pos = None
            degree = None
            if pos_xy:
                pos = self.gt.vertex_properties['pos']
            if deg:
                degree = self.gt.degree_property_map("total")

            graph_draw(self.gt, pos=pos, vertex_size=degree, vorder=degree,
                       vertex_fill_color=degree, edge_color='black',
                       output=self.output_dir + "/" + fname + '.png')
            print("%s saved" % fname)


        if len(self.networkx_plot) > 0:
            print("Plotting graph in networkx ...")

        for plot in self.networkx_plot:
            pos_xy = plot.get('pos', False)
            deg = plot.get('degree', False)
            fname = plot.get('fname', 'graph_nx')

            pos = None
            degree = None
            node_size = 250
            node_color = 'cyan'
            if pos_xy:
                x = dict(self.gnx.nodes(data='x'))
                keys = x.keys()
                y = dict(self.gnx.nodes(data='y'))
                values = zip(x.values(), y.values())
                pos = dict(zip(keys, values))
            if deg:
                degree = dict(self.gnx.degree)
                node_size = list([v*50 for v in degree.values()])
                node_color = list([v*50 for v in degree.values()])

            nx.draw(self.gnx, nodelist=list(self.gnx.nodes()),
                    node_size=node_size, pos=pos, node_color=node_color)
            plt.savefig(self.output_dir + "/" + fname)
            print("%s saved" % fname)

    def iter_analysis_types(self):
        """Iterate through all the analysis specified."""
        for plot in self.analysis_plots:

            analysis_type = plot.get('analysis_type')
            possible_an = ['similarity_all','sim_exclude_neurons', 'sim_exclude_ctype', 'sim_celltype'\
                           'sim_list_neurons', 'pagerank', 'betweeness','closeness', 'degree_dist'\
                           'weigths_dist', 'clustering']
            assert analysis_type in possible_an

            similarity = plot.get('similarity', None)



    def _generate_igraph(self, A):
        """Generate igraph to compute similarity."""
        G = ig.Graph.Adjacency((A > 0).tolist())
        G.es['weight'] = A[A.nonzero()]
        G.vs['label'] = list(self.gnx.nodes())

        return G

    def jaccard_similarity(self, A, store=False):

        G = self._generate_igraph(A)
        sim_mat = np.matrix(G.similarity_jaccard())

        fig = plt.figure(figsize=(16,10))
        ax = fig.add_subplot(111)
        i = ax.imshow(sim_mat)
        ax.set_xticks(np.arange(self.gnx.number_of_nodes()))
        ax.set_xticklabels(self.gnx.nodes(),rotation=90)
        ax.set_yticks(np.arange(self.gnx.number_of_nodes()))
        ax.set_yticklabels(self.gnx.nodes())
        plt.colorbar(i, ax=ax)
        plt.savefig()
        if store:
            self.sim_mat = sim_mat


if __name__ == '__main__':

    assert len(sys.argv) == 2
    config_f = sys.argv[1]

    ga = GraphAnalysis(config_f)
    ga.plot_graphs()


    # similarity, plus plots with average
    # sort similarity -> convert similarity mat into a nx graph and run algorithm

    # removing single cell
    # removing cell type : configs file
    # save output using nameofexcluded_cell

    # connection probability

    # print("Fitting linear function between Adjacency matrix (A) and Jaccard similarity (J)")
    # if m > 0 :
    #     print("Increasing trend between A and J")
    # else:
    #     print("Decreasing trend between A and J")

    # pearson correlation

    # spectral analysis pltos

    # page rank/betweeness/closeness + plot

    # statistics : degrees + weights

    # clustering

    # motifs
