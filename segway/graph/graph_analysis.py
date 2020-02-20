import numpy as np
import math
import networkx as nx
from graph_tool.all import *
import igraph as ig
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import itertools
import scipy
import matplotlib as mpl
from matplotlib import cm
import graph_tool_functions as gtf
from segway.graph.neuron_graph import NeuronGraph



class GraphAnalysis():
    """GraphAnalysis """
    def __init__(self, config_file):
        super(GraphAnalysis, self).__init__()
        self.config_file = config_file
        self.__initialize_configs(config_file)

    def __initialize_configs(self, input_file):
        """Initialize default values."""
        # create output directory with same name of config file
        self.directory = (self.config_file[:-5])  # exclude format
        self.overwrite = False
        self.similarity = False
        self.ave_similarity = False

        # save output files names if analysis specified in config file

        self.output_similarity = self.directory + '/similarity_mat.png'

    ##### COPY/PASTA : TO MODIFY
    def __read_configs(self, input_file):
        """Recursively read configs from given JSON file."""
        logger.info("Parsing graph from config %s" % input_file)
        with open(input_file) as f:
            params = json.load(f)

        assert 'input_config_files' in params
        self.g = SynapseGraph(params['input_config_files'])  # NeuronGraph

        for key in params:
            # if key == 'debug_edges_list':
            #     print(self.debug_edges_list)
            #     print(params[key])
            if hasattr(self, key) and isinstance(getattr(self, key), list):
                # extend if parameter is a list
                assert isinstance(params[key], list)
                # setattr(self, key, getattr(self, key).extend(params[key]))
                getattr(self, key).extend(params[key])
            else:
                # initialize or overwrite value
                setattr(self, key, params[key])

if __name__ == '__main__':
    config_f = sys.argv[1]
    with open(config_f) as f:
        config = json.load(f)

    g = NeuronGraph(config_f)

    # convert graph into graph-tool
    gt, neurons_to_ids = gtf.create_nodes_and_attr_gt(g.get_graph())
    print("Number of nodes: ", gt.num_vertices())
    edge_list_ids = gtf.convert_el_to_ids(g.edge_list, neurons_to_ids)
    gt = gtf.create_edges_graph_gt(gt, edge_list_ids)
    # self.output_graph_path = self.directory + '/output_graph.gpickle'
    graph_draw(gt, output=g.directory + '/graph_gt.png')
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
