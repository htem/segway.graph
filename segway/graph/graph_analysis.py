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

logger = logging.getLogger(__name__)

class GraphAnalysis():
    """GraphAnalysis """
    def __init__(self, config_file):
        super(GraphAnalysis, self).__init__()
        self.config_file = config_file

        self.__initialize_configs()
        self.__read_configs()

        os.makedirs(self.output_dir, exist_ok=True)

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
        with open(self.config_file) as f:
            params = json.load(f)

        assert 'graph_config_file' in params
        # read existing graph networkx
        neuronGraph = SynapseGraph(params['graph_config_file'])
        logger.info("Parsing graph from config %s" % params['graph_config_file'])
        self.gnx = neuronGraph.get_graph()
        print("### Info : read networkx graph ", self.gnx)
        self.neuron_list = neuronGraph.get_neurons_list()
        self.edge_list = neuronGraph.edge_list
        self.edge_list_df = neuronGraph.edge_list_df
        self.A = neuronGraph.get_matrix()

        for key in params:
            if hasattr(self, key) and isinstance(getattr(self, key), list):
                # extend if parameter is a list
                assert isinstance(params[key], list)
                # setattr(self, key, getattr(self, key).extend(params[key]))
                getattr(self, key).extend(params[key])
            else:
                # initialize or overwrite value
                setattr(self, key, params[key])


if __name__ == '__main__':

    assert len(sys.argv) == 2
    config_f = sys.argv[1]

    ga = GraphAnalysis(config_f)

    # # convert graph into graph-tool
    # gt, neurons_to_ids = gtf.create_nodes_and_attr_gt(g.get_graph())
    # print("Number of nodes: ", gt.num_vertices())
    # edge_list_ids = gtf.convert_el_to_ids(g.edge_list, neurons_to_ids)
    # gt = gtf.create_edges_graph_gt(gt, edge_list_ids)
    # # self.output_graph_path = self.directory + '/output_graph.gpickle'
    # graph_draw(gt, output=g.directory + '/graph_gt.png')


    # PLOT NX, PLOT POSITIONS OR NOT (CHECK JUPYTER)

    # degree (option in config file)

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
