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

        self.ctype_idx = self.create_dict_ctype_idx()

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
        self.similarity_plots = []
        self.general_analysis = []
        self.count_motifs = []
        self.random_motifs = []
        self.plot_subgraphs = []
        self.plot_motifs_counts = []
        self.identify_motifs = {}
        # useful dictionaries that maps the variables needed
        self.map_counts = defaultdict(list)
        self.map_motifs = defaultdict(list)


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
        self.cell_types_dict = nx.get_node_attributes(self.gnx, 'cell_type')
        self.cell_types = list(set(list(self.cell_types_dict.values())))

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
        # create reverse dictionary
        self.ids_to_neurons = gtf.create_ids_to_neurons(self.neurons_to_ids)
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

    def create_dict_ctype_idx(self):

        index = defaultdict(list)

        for ct in self.cell_types:
            for i, val in enumerate(self.cell_types_dict.items()):
                if ct == val[1]:
                    index[ct].append(i)

        return index

    def iter_similarity_plots(self):
        """Iterate through all the analysis specified."""
        # Possible analysis
        possible_an = ['similarity_all','sim_exclude_neurons', 'sim_exclude_ctype', 'similarity_zoom']

        for plot in self.similarity_plots:

            similarity_plot = plot.get('analysis_type')
            fit_probability = plot.get('fit_probability', False)
            assert similarity_plot in possible_an
            pick_neurons = plot.get('neurlist_to_ex', [])
            self.zoom_neurons_list = plot.get('neurons_list', [])
            ctype_to_ex = plot.get('ctype_to_ex', [])
            fname = similarity_plot
            store = False

            if similarity_plot == 'similarity_all':
                store = True

            if similarity_plot == 'sim_exclude_ctype':
                pick_neurons = [name for name, ct in self.cell_types_dict.items() if ct == ctype_to_ex]
                fname = fname + "_" + ctype_to_ex

            to_zero = [i for p in pick_neurons for i, n in enumerate(self.neuron_list) if p == n]
            A = self.A.copy()
            A[:, to_zero] = 0
            A[to_zero, :] = 0

            if similarity_plot == 'similarity_zoom':
                idx = []
                for n in self.zoom_neurons_list:
                    idx.append(self.neurons_to_ids[n])

                A = self.A[idx,:][:,idx]

            self.jaccard_similarity(A, fname, store=store)

            if fit_probability:
                print("### Info: Fitting linear function between Adjacency matrix (A) and Jaccard similarity (J)")
                print("y = mx + b [y = A, x = J]")
                x = np.array(self.sim_mat).reshape(len(self.sim_mat)**2)
                y = np.array(self.A).reshape(len(self.A)**2)

                m, b = np.polyfit(x, y, 1)
                print("m = %f , b = %f " % (m,b))
                if m > 0 :
                    print("Increasing trend between A and J")
                else:
                    print("Decreasing trend between A and J")

                print("### Info: Pearson correlation ...")
                corr, pv = scipy.stats.pearsonr(x,y)
                print("Pearson corr = " , corr)
                if pv < 0.01:
                    print("p-value < 0.01 : Pearson correlation is significant!")
                else:
                    print("Nothing can be said on the correlation of the two variables!")

                fig = plt.figure(figsize=(13,10))
                R = scipy.signal.correlate2d(self.sim_mat,self.A, mode='same')
                # R[np.isnan(R)] = 0
                plt.imshow(R)
                plt.colorbar()
                plt.title("correlation between A and J")
                fig.savefig(self.output_dir + "/" + "corr_sim_adj")

    def _generate_igraph(self, A):
        """Generate igraph to compute similarity."""
        G = ig.Graph.Adjacency((A > 0).tolist())
        G.es['weight'] = A[A.nonzero()]
        G.vs['label'] = self.neuron_list

        return G

    def jaccard_similarity(self, A, fname, store=False):

        G = self._generate_igraph(A)
        sim_mat = np.matrix(G.similarity_jaccard())

        fig = plt.figure(figsize=(16,10))
        ax = fig.add_subplot(111)
        i = ax.imshow(sim_mat)
        if fname == 'similarity_zoom':
            ax.set_xticks(np.arange(len(self.zoom_neurons_list)))
            ax.set_xticklabels(self.zoom_neurons_list, rotation=90)
            ax.set_yticks(np.arange(len(self.zoom_neurons_list)))
            ax.set_yticklabels(self.zoom_neurons_list)
        else:
            ax.set_xticks(np.arange(self.gnx.number_of_nodes()))
            ax.set_xticklabels(self.neuron_list, rotation=90)
            ax.set_yticks(np.arange(self.gnx.number_of_nodes()))
            ax.set_yticklabels(self.neuron_list)

        plt.colorbar(i, ax=ax)
        if store:
            self.sim_mat = sim_mat
        plt.savefig(self.output_dir + "/" + fname)
        print("%s saved" % fname)


    def std_graph_analysis(self):

        # Possible analysis
        possible_an = ['pagerank',
                       'betweeness',
                       'closeness',
                       'degree_dist',
                       'weigths_dist',
                       'clustering']

        for an in self.general_analysis:
            analysis_type = an.get('analysis_type')
            assert analysis_type in possible_an

            colorbar = an.get('colorbar', False)
            self.ctype = an.get('ctype', False)  # it will plot the results wrt the cell types
            clust_type = an.get('clust_type', [])
            random = an.get('random', False)
            degree_type = an.get('degree_type', [])

            if random:
                # generate another plot for random
                self.rg = Graph(self.gt)
                graph_tool.generation.random_rewire(self.rg, model='erdos', n_iter=100)
            if analysis_type == 'pagerank':
                pr_res = self.pagerank(random=random)
                if colorbar:
                    self.plot_colorbar(pr_res.a, 'pagerank')
            elif analysis_type == 'betweeness':
                bet_res = self.betweeness(random=random)
                if colorbar:
                    self.plot_colorbar(bet_res.a, 'betweeness')
            elif analysis_type == 'closeness':
                clo_res = self.closeness(random=random)
                if colorbar:
                    self.plot_colorbar(clo_res.a, 'closeness')
            elif analysis_type == 'degree_dist':
                for deg in degree_type:
                    self.degree_histogram(deg, random=random)
            elif analysis_type == 'weigths_dist':
                self.weights_histogram()
            elif analysis_type == 'clustering':
                for cl_type in clust_type:
                    self.clustering(cl_type, random=random)


    def clustering(self, cl_type, random=False):

        if cl_type == 'global':
            gc = graph_tool.clustering.global_clustering(self.gt)
            print("Global clustering coefficient of the graph:")
            print("value = %f, std = %f" %(gc[0], gc[1]))
            if random:
                gcr = graph_tool.clustering.global_clustering(self.rg)
                print("Global clustering coefficient of a RANDOM graph:")
                print("value = %f, std = %f" %(gcr[0], gcr[1]))

        elif cl_type == 'local':
            loc_clust = graph_tool.clustering.local_clustering(self.gt)

            fig = plt.figure(figsize=(12,12))
            ax = fig.add_subplot(111)
            ax.scatter(self.ids_to_neurons.keys(),loc_clust.a)
            ax.grid(True)
            ax.set_xticks(list(self.ids_to_neurons.keys()))
            ax.set_xticklabels(self.ids_to_neurons.values(), rotation=90, fontsize=12)
            ax.set_ylabel("Local clustering coeff")

            if random:
                loc_clust_r = graph_tool.clustering.local_clustering(self.rg)
                # scatter in red
                ax.scatter(self.ids_to_neurons.keys(), loc_clust_r.a, c='r')
                ax.legend(['connectome', 'random'])


            # adding red lines to highlight outliers
            M = np.argmax(loc_clust.a)
            m = np.argmin(loc_clust.a)
            xposition = [M,m]
            yposition = list(loc_clust.a[xposition])
            for xc in xposition:
                plt.axvline(x=xc, color='r', linestyle='--')

            for yc in yposition:
                plt.axhline(y=yc, color='r', linestyle='--')

            fig.savefig(self.output_dir + "/" + "local_clustering")


    def weights_histogram(self, random=False):

        edhist = graph_tool.stats.edge_hist(self.gt,
                                            self.gt.edge_properties["weight"])
        fig = plt.figure(figsize=(16,10))

        if random:
            edhist_r = graph_tool.stats.edge_hist(self.rg,
                                                  self.rg.edge_properties["weight"])

            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax2.hist(edhist_r[1][:-1], edhist_r[1], weights=edhist_r[0])
            ax2.set_xlabel("Random weights")
            ax2.set_ylabel("Random counts")
        else:
            ax1 = fig.add_subplot(111)

        ax1.hist(edhist[1][:-1], edhist[1], weights=edhist[0])
        ax1.set_xlabel("Weights")
        ax1.set_ylabel("Counts")
        fig.savefig(self.output_dir + "/" + "weights_hist")

    def degree_histogram(self, deg_t, random=False):

        deg = self.gt.degree_property_map(deg_t)
        h = graph_tool.stats.vertex_hist(self.gt, deg)

        fig = plt.figure(figsize=(16,10))

        if random:
            deg_r = self.gt.degree_property_map(deg_t)
            h_r = graph_tool.stats.vertex_hist(self.rg, deg)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax2.hist(h_r[1][:-1], h_r[1], weights=h_r[0])
            ax2.set_xlabel("random " + deg_t + " degrees")
            ax2.set_ylabel("random counts")
        else:
            ax1 = fig.add_subplot(111)

        ax1.hist(h[1][:-1], h[1], weights=h[0])
        ax1.set_xlabel(deg_t + " degrees")
        ax1.set_ylabel("counts")
        fname = "degrees_hist"
        if deg_t == 'in':
            fname = fname + "_in"
        elif deg_t == 'out':
            fname = fname + "_out"
        elif deg_t == 'all':
            fname = fname + "_all"

        fig.savefig(self.output_dir + "/" + fname)

    def closeness(self, random=False):

        fname = 'closeness.png'
        print('### Info: computing closeness centrality on graph ...')
        c = closeness(self.gt)
        c.a[np.isnan(c.a)] = 0

        if self.ctype:
            vertex_text= self.gt.vertex_properties['cell_type']
        else:
            vertex_text = self.gt.vertex_properties['node_name']

        graph_draw(self.gt, vertex_fill_color=c,
                   vertex_text=vertex_text,
                   vcmap=cm.gist_heat,
                   vorder=c, output=self.output_dir + "/" + fname)

        if random:
            print('### Info: computing closeness centrality on RANDOM graph ...')
            r_fname = 'random_closeness.png'
            c = closeness(self.rg)
            c.a[np.isnan(c.a)] = 0
            graph_draw(self.gt, vertex_fill_color=c,
                       vertex_text=vertex_text,
                       vcmap=cm.gist_heat,
                       vorder=c, output=self.output_dir + "/" + r_fname)

        return c

    def betweeness(self, random=False):

        fname = 'betweeness.png'
        print('### Info: computing betweeness centrality on graph ...')
        vp, ep = betweenness(self.gt)

        if self.ctype:
            vertex_text= self.gt.vertex_properties['cell_type']
        else:
            vertex_text = self.gt.vertex_properties['node_name']

        graph_draw(self.gt, vertex_fill_color=vp,
                   vertex_text=vertex_text,
                   edge_pen_width=prop_to_size(ep, mi=0.5, ma=5),
                   vcmap=cm.gist_heat,
                   vorder=vp,
                   output=self.output_dir + "/" + fname)
        if random:
            print('### Info: computing betweeness centrality on RANDOM graph ...')
            r_fname = 'random_betweeness.png'
            vp, ep = betweenness(self.rg)
            graph_draw(self.gt, vertex_fill_color=vp,
                       vertex_text=vertex_text,
                       edge_pen_width=prop_to_size(ep, mi=0.5, ma=5),
                       vcmap=cm.gist_heat,
                       vorder=vp,
                       output=self.output_dir + "/" + r_fname)

        return vp

    def pagerank(self, random=False):

        fname = 'pagerank.png'
        print("### Info: Running PageRank on graph ...")
        pr_res = graph_tool.centrality.pagerank(self.gt)
        pr_list = [(i,pr) for i, pr in enumerate(pr_res)]
        pr = sorted(pr_list, key = lambda x: x[1], reverse=True)
        self.sorted_pr = [self.ids_to_neurons[tup[0]] for tup in pr]
        print("PageRank results sorted by score:")
        print(self.sorted_pr)

        if self.ctype:
            vertex_text= self.gt.vertex_properties['cell_type']
        else:
            vertex_text = self.gt.vertex_properties['node_name']

        graph_draw(self.gt, vertex_text=vertex_text, vorder=pr_res,
                   vcmap=cm.gist_heat, vertex_fill_color=pr_res,
                   output=self.output_dir + "/" + fname)
        if random:
            print("### Info: Running PageRank on random graph ...")
            r_fname = 'random_pagerank.png'
            pr_res = graph_tool.centrality.pagerank(self.rg)
            pr_list = [(i,pr) for i, pr in enumerate(pr_res)]
            pr = sorted(pr_list, key = lambda x: x[1], reverse=True)
            self.sorted_pr = [self.ids_to_neurons[tup[0]] for tup in pr]
            print("PageRank results for RANDOM graph sorted by score:")
            print(self.sorted_pr)

            graph_draw(self.rg, vertex_text=vertex_text, vorder=pr_res,
                       vcmap=cm.gist_heat, vertex_fill_color=pr_res,
                       output=self.output_dir + "/" + r_fname)

        return pr_res

    def plot_colorbar(self, array, label):
        # Make a figure and axes with dimensions as desired.
        fig = plt.figure(figsize=(9, 3))
        ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])

        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        norm = mpl.colors.Normalize(vmin=min(array), vmax=max(array))

        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cm.gist_heat,
                                        norm=norm,
                                        orientation='horizontal')
        cb1.set_label(label)
        plt.savefig(self.output_dir + "/" + label + "_colorbar")

    def countfind_motifs(self, gt, motif, random=False):

        if motif == 'duplets':
            dup_motifs, dup_counts = graph_tool.clustering.motifs(gt,2)
            if random:
                print("*** Random Graph ***")
                self.r_dup_motifs = dup_motifs
                self.r_dup_counts = dup_counts
                self.map_counts[motif].append(dup_counts)
            else:
                print("*** Connectome ***")
                self.dup_motifs = dup_motifs
                self.dup_counts = dup_counts
                self.map_counts[motif].append(dup_counts)
                self.map_motifs[motif].append(dup_motifs)

            print("### Info : found %d duplets" % len(dup_motifs))

        elif motif == 'triplets':
            tri_motifs, tri_counts = graph_tool.clustering.motifs(gt,3)
            if random:
                print("*** Random Graph ***")
                self.r_tri_motifs = tri_motifs
                self.r_tri_counts = tri_counts
                self.map_counts[motif].append(tri_counts)
            else:
                print("*** Connectome ***")
                self.tri_motifs = tri_motifs
                self.tri_counts = tri_counts
                self.map_counts[motif].append(tri_counts)
                self.map_motifs[motif].append(tri_motifs)

            print("### Info : found %d triplets" % len(tri_motifs))

        elif motif == 'quadruplets':
            quad_motifs, quad_counts = graph_tool.clustering.motifs(gt,4)
            if random:
                print("*** Random Graph ***")
                self.r_quad_motifs = quad_motifs
                self.r_quad_counts = quad_counts
                self.map_counts[motif].append(quad_counts)
            else:
                print("*** Connectome ***")
                self.quad_motifs = quad_motifs
                self.quad_counts = quad_counts
                self.map_counts[motif].append(quad_counts)
                self.map_motifs[motif].append(quad_motifs)

            print("### Info : found %d quadruplets" % len(quad_motifs))

        elif motif == 'quintuplets':
            quin_motifs, quin_counts = graph_tool.clustering.motifs(gt,5)
            if random:
                print("*** Random Graph ***")
                self.r_quin_motifs = quin_motifs
                self.r_quin_counts = quin_counts
                self.map_counts[motif].append(quin_counts)
            else:
                print("*** Connectome ***")
                self.quin_motifs = quin_motifs
                self.quin_counts = quin_counts
                self.map_counts[motif].append(quin_counts)
                self.map_motifs[motif].append(quin_motifs)

            print("### Info : found %d quintuplets" % len(quin_motifs))

        else:
            print("Error: only duplets, triplets, quadruplets and quintuplets have been implemented!")

    def plot_counts_motifs(self):

        # create random graph if not existing
        if not hasattr(self, 'rg'):
            self.rg = Graph(self.gt)
            graph_tool.generation.random_rewire(self.rg, model='erdos', n_iter=100)

        # map_counts = {
        #            'duplets' : [self.dup_counts, self.r_dup_counts],
        #            'triplets': [self.tri_counts, self.r_tri_counts],
        #            'quadruplets': [self.quad_counts, self.r_quad_counts],
        #            'quintuplets': [self.quin_counts, self.r_quin_counts]
        #            }

        for i, plots in enumerate(self.count_motifs):
            for motif, motif_idx in plots.items():
                self.countfind_motifs(self.gt, motif)
                counts = np.array(self.map_counts[motif][0])[np.array(motif_idx)]
                fig = plt.figure(figsize=(14,10))
                plt.scatter(np.linspace(0,len(counts)-1,len(counts)), counts)
                plt.xticks(np.linspace(0,len(counts)-1,len(counts)))

                if motif in self.random_motifs:
                    self.countfind_motifs(self.rg, motif, random=True)
                    # check if the number of motifs is the same, if not append 0 for not found motifs
                    var = self.map_counts[motif]
                    numdiff = abs(len(var[0]) - len(var[1]))
                    if numdiff != 0:
                        idx = np.argmin([len(var[0]), len(var[1])])
                        for i in range(numdiff):
                            var[idx].append(0)
                    # plot also random
                    rcounts = np.array(self.map_counts[motif][1])[np.array(motif_idx)]
                    plt.scatter(np.linspace(0,len(rcounts)-1,len(rcounts)), rcounts, c='r')
                    plt.legend(['connectome', 'random'])

                plt.plot(counts, color='black', linestyle='--', alpha=0.6)
                fig.savefig(self.output_dir + "/" + motif+ "_motifs"+str(i))


    def plot_motifs(self):

        for mtype in self.plot_subgraphs:
            if mtype == 'duplets':
                outdir = '/Duplets_motifs'
                out_path = self.output_dir + outdir
                if not os.path.isdir(out_path):
                    print("### Info : plotting all duplets motifs ...")
                    os.makedirs(out_path)
                    # fill the directory, otherwise no need to save all the motifs
                    for i, motif in enumerate(self.tri_motifs):
                        graph_draw(motif, output=out_path+"/motif_"+str(i)+".png")

            elif mtype == 'triplets':
                outdir = '/Triplets_motifs'
                out_path = self.output_dir + outdir
                if not os.path.isdir(out_path):
                    print("### Info : plotting all triplets motifs ...")
                    os.makedirs(out_path)
                    # fill the directory, otherwise no need to save all the motifs
                    for i, motif in enumerate(self.tri_motifs):
                        graph_draw(motif, output=out_path+"/motif_"+str(i)+".png")

            elif mtype == 'quadruplets':
                outdir = '/Quadruplets_motifs'
                out_path = self.output_dir + outdir
                if not os.path.isdir(out_path):
                    print("### Info : plotting all quadruplets motifs ...")
                    os.makedirs(out_path)
                    for i, motif in enumerate(self.quad_motifs):
                        graph_draw(motif, output=out_path+"/motif_"+str(i)+".png")

            elif mtype == 'quintuplets':
                outdir = '/Quintuplets_motifs'
                out_path = self.output_dir + outdir
                if not os.path.isdir(out_path):
                    print("### Info : plotting all quintuplets motifs ... WARNING: This will take a while!")
                    os.makedirs(out_path)
                    for i, motif in enumerate(self.quin_motifs):
                        graph_draw(motif, output=out_path+"/motif_"+str(i)+".png")

    def find_neurons_motifs(self, key, value):

        # useful dictionary
        num_name = {"duplets" : 2,
                    "triplets": 3,
                    "quadruplets": 4,
                    "quintuplets": 5}
        print("-------------------------------------")
        print("Identified %s type %d : " % (key, value))

        vps = graph_tool.topology.subgraph_isomorphism(self.map_motifs[key][0][value], self.gt, induced=True, max_n = 100)

        m = []
        for j in range(len(vps)):
            a = np.zeros(num_name[key], dtype=int)
            for i, x in enumerate(vps[j]):
                a[i] = x

            add = np.sort(a)
            m.append(tuple(add))

        identified = list(set(m))

        # convert ids into neuron name
        n_identified = []
        for i in identified:
            neurons = []
            for n_id in i:
                neurons.append(self.ids_to_neurons[n_id])
            n_identified.append(neurons)

        print(n_identified)

    def print_neurons_motifs(self):

        # run store counts only if not existing and only with the keys specified in the identify_motifs key config
        # itero nelle keys,

        # self.map_motifs = {"duplets" : motifs}

        if len(self.map_motifs) == 0:
            for key, value in self.identify_motifs.items():
                self.countfind_motifs(self.gt, key)
            # run countmotifs iterando nelle keys
        else:
            # the motifs to identify need to be stored
            assert set(list(self.identify_motifs.keys())) < set(list(self.map_counts.keys()))

        for key, values in self.identify_motifs.items():
            if isinstance(values, list):
                for value in values:
                    self.find_neurons_motifs(key,value)
            else:
                assert isinstance(values, int)
                self.find_neurons_motifs(key,values)

        return

if __name__ == '__main__':

    assert len(sys.argv) > 1
    config_f = sys.argv[1]
    ga = GraphAnalysis(config_f)

    analysis = False
    print_motifs = False
    if len(sys.argv) == 3:
        var = sys.argv[2]
        if var == '--all':
            analysis = True
            print_motifs = True
        elif var == '--ga':
            # run graph analysis
            analysis = True
        elif var == '--pm':
            print_motifs = True
        else:
            print("ERROR: third argument must be in [--all, --ga, --pm]")
            exit()

    if analysis:
        ga.plot_graphs()
        ga.std_graph_analysis()
        ga.plot_counts_motifs()
        ga.plot_motifs()

    if print_motifs:
        ga.print_neurons_motifs()
