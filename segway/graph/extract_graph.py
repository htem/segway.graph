import numpy as np
import networkx as nx
import logging
import os
import sys
import json
import time
from collections import defaultdict

import pandas as pd

from database_synapses import SynapseDatabase
from database_superfragments import SuperFragmentDatabase
sys.path.insert(0, '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/repos/funlib.show.neuroglancer')
sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/segwaytool.proofreading')
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway/synful_tasks')
import segwaytool.proofreading
import segwaytool.proofreading.neuron_db_server
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ExtractGraph():
    """ExtractGraph allows the creation of the graph.

    Also outputs and plots are generated.
    """

    def __init__(self, config_file):
        """Initialize attributes."""
        self.config_file = config_file

        self.__initialize_configs(config_file)
        self.__read_configs(config_file)
        self.__check_configs()

        os.makedirs(self.directory, exist_ok=True)

        self.__connect_DBs()
        self.create_graph()
        self.preprocess_graph()

        self.A = nx.to_numpy_matrix(self.g)
        self.debug_spec_edges()  # debug specified edges

        # iterate over the plots configs
        for plot in self.plots:
            self.plot_adj_mat(plot)

    def __initialize_configs(self, input_file):
        '''Initialize default values'''
        # create output directory with same name of config file
        self.directory = (self.config_file[:-5])  # exclude format
        self.overwrite = False
        self.add_edge_list = []
        self.exclude_neurons = []
        self.tags_to_exclude = []
        self.exclude_edges = []
        self.exclude_synapses = []
        self.debug_edges = False
        self.debug_edges_list = []
        self.rename_rules = []
        self.plots = []
        self.weights_with_dist = False

        # save output files names
        self.output_graph_path = self.directory + '/output_graph.gpickle'
        self.output_edges_path = self.directory + '/output_edges.csv'
        # save if existing in config
        self.output_graph_pp_path = self.directory + '/output_graph_pp.gpickle'
        self.output_debug_edges_path = self.directory + '/output_debug_edges.csv'

    def __check_configs(self, input_file):

        # make sure that essential params are defined
        for p in [
                'db_name', 'db_host', 'db_name_n', 'input_roi_offset',
                'input_roi_size', 'voxel_size_xyz', 'syn_score_threshold',
                'input_method', 'mode_weights',
                ]:
            assert hasattr(self, p)

    def __read_configs(self, input_file):
        '''Recursively read configs from given JSON file'''
        logger.info("Parsing configuration from %s" % input_file)
        with open(self.config_file) as f:
            params = json.load(f)

        if 'input_config_files' in params:
            if isinstance(params['input_config_files'], list):
                for f in params['input_config_files']:
                    self.__read_configs(f)
            else:
                self.__read_configs(params['input_config_files'])

        for key in params:
            if hasattr(self, key) and isinstance(getattr(self, key), list):
                # extend if parameter is a list
                assert isinstance(params[key], list)
                setattr(self, key, getattr(self, key).extend(params[key]))
            else:
                # initialize or overwrite value
                setattr(self, key, params[key])

    def __connect_DBs(self):

        syn_db = SynapseDatabase(
            db_name=self.db_name, db_host=self.db_host,
            db_col_name='synapses',)

        sf_db = SuperFragmentDatabase(
            db_name=self.db_name,
            db_host=self.db_host,
            db_col_name='superfragments',)

        neuron_db = segwaytool.proofreading.neuron_db_server.NeuronDBServer(
                    db_name=self.db_name_n,
                    host=self.db_host,)
        neuron_db.connect()

        self.syn_db = syn_db
        self.sf_db = sf_db
        self.neuron_db = neuron_db

    def _get_neurons_info_db(self):

        nodes_attr = {}
        for nid in self.neurons_list:
            neuron = self.neuron_db.get_neuron(nid).to_json()
            # create dictionary with attributes per neuron
            idict = dict()
            idict['cell_type'] = neuron['cell_type']
            idict['x'] = neuron['soma_loc']['x']
            idict['y'] = neuron['soma_loc']['y']
            idict['z'] = neuron['soma_loc']['z']
            idict['tags'] = neuron['tags']
            idict['finished'] = neuron['finished']
            # assign attributes dictionary
            nodes_attr[nid] = idict

        return nodes_attr

    def create_nodes_graph(self):
        """Create nodes of networkx graph."""
        G = nx.DiGraph()
        for i, n in enumerate(self.neurons_list):
            G.add_node(n)

        nx.set_node_attributes(G, self.nodes_attr)
        print("### Info : Number of nodes in the graph : ", G.number_of_nodes())

        return G

    def create_neurons_dict_sf(self):
        """Dictionary with neurons as keys and their sf as values."""
        neurons_dict_sf = dict()

        for nid in self.neurons_list:
            superfragments = self.neuron_db.get_neuron(nid).to_json()['segments']
            sfs_list = [int(item) for item in superfragments]
            neurons_dict_sf[nid] = sfs_list

        return neurons_dict_sf

    def create_sf_dict_neurons(self):
        """Create reverse dictionary (of the above)."""
        sf_to_neurons = dict()
        for nid in self.neurons_list:
            for sfid in self.neurons_dict_sf[nid]:
                sf_to_neurons[sfid] = nid

        return sf_to_neurons

    def _get_superfragments_info_db(self, superfragments_list):

        sfs_dict = self.sf_db.read_superfragments(sf_ids=superfragments_list)

        return sfs_dict

    def create_edges_list(self):
        """Create edges and synapses lsit."""
        neuron_list = np.array(list(self.neurons_dict_sf.keys()))  # all neurons
        edge_list = set()
        synapse_list = set()

        for nid in neuron_list:
            # for each neuron, we get their post partners as sf
            # convert them to neuron_id and add directed edge

            sfs_dict = self._get_superfragments_info_db(self.neurons_dict_sf[nid])
            for sf in sfs_dict:

                post_partners_sf = sf['post_partners']
                # print("post_partners_sf:", post_partners_sf)
                for post_sf in post_partners_sf:
                    if post_sf not in self.sf_to_neurons:
                        # post neuron not in input list
                        continue
                    post_neuron = self.sf_to_neurons[post_sf]
                    if post_neuron != nid:
                        edge_list.add((nid, post_neuron))

                pre_partners_sf = sf['pre_partners']
                # print("pre_partners_sf:", pre_partners_sf)
                for pre_sf in pre_partners_sf:
                    if pre_sf not in self.sf_to_neurons:
                        # post neuron not in input list
                        continue
                    pre_neuron = self.sf_to_neurons[pre_sf]
                    if pre_neuron != nid:
                        edge_list.add((pre_neuron, nid))

                synapse_list.update(sf['syn_ids'])

        edge_list = list(edge_list)
        return edge_list, synapse_list

    def create_syns_dict(self):
        """Given the list of synapses : a dictionary is created as query result."""
        # get synapse attributes
        print("###: Info: querying synapses DB")
        start = time.time()
        query = {'$and': [{'id': {'$in': list(self.synapse_list)}},
                {'score': {'$gt': self.syn_score_threshold}}]}

        # query = {'id' : { '$in' : list(synapse_list) }}
        synapses_query = self.syn_db.synapses.find(query)
        syns_dict = defaultdict(lambda: defaultdict(dict))

        # {id: {'syn_loc': [x,y,z], 'area': 123, 'length': 0.5}}
        for i, syn in enumerate(synapses_query):
            # take location to store synapses length/distance
            pre = np.array([syn['pre_x'], syn['pre_y'], syn['pre_z']])
            post = np.array([syn['post_x'], syn['post_y'], syn['post_z']])

            syns_dict[i] = {'syn_loc': [int(syn['x'] / self.voxel_size_xyz[0]),
                                        int(syn['y'] / self.voxel_size_xyz[1]),
                                        int(syn['z'] / self.voxel_size_xyz[2])],
                            # 'area': syn['area'],
                            'dist': np.linalg.norm(pre - post),
                            'sf_pre': syn['id_superfrag_pre'],
                            'sf_post': syn['id_superfrag_post']
                            }

        print("Synapses query and dict creation took %f s" % (time.time() - start))

        return syns_dict

    def compute_weights(self):
        """
        Compute weights according to the area of the synapse.

        if area is True the area of the synapses is taken into consideration
        if dist is True, the distance from the soma is also considered
        if area and dist are False the weights will be the number of synapses
        mode = "count"/"area"
        """
        mode_area = self.mode_weights == "area"
        mode_length = self.mode_weights == "length"

        print("## Info : Computing the weights of the graph...")
        start = time.time()

        syn_weights = defaultdict(float)

        synapses_dict = defaultdict(list)

        for k, syn in self.syns_dict.items():
            pre_neuron = syn['sf_pre']
            post_neuron = syn['sf_post']
            if pre_neuron not in self.sf_to_neurons or post_neuron not in self.sf_to_neurons:
                continue

            pre_neuron = self.sf_to_neurons[pre_neuron]
            post_neuron = self.sf_to_neurons[post_neuron]

            if pre_neuron == post_neuron:
                continue

            weight = 1
            if mode_area:
                weight = syn['area'] / 1e+3
            if mode_length:
                weight = syn['dist']
            if self.weights_with_dist:
                soma_loc = np.array([self.g.nodes[post_neuron]['x'],
                                     self.g.nodes[post_neuron]['y'],
                                     self.g.nodes[post_neuron]['z']])

                syn_loc = np.array([syn['syn_loc']])
                distance = np.linalg.norm(soma_loc - syn_loc)
                weight = weight / distance

            syn_weights[(pre_neuron, post_neuron)] += weight
            synapses_dict[(pre_neuron, post_neuron)].append(syn['syn_loc'])

        weights = []
        synapses_locs = []
        filt_edge_list = self.edge_list.copy()

        for e in self.edge_list:
            if e in syn_weights:
                weights.append(syn_weights[e])
                synapses_locs.append(synapses_dict[e])

            elif e not in syn_weights:
                print("Edge %s not found in synapse attributes" % str(e))
                filt_edge_list.remove(e)
                # continue
            # assert e in syn_weights
            # weights.append(syn_weights[e])

        print("Weights creation took %f s" % (time.time() - start))

        return weights, filt_edge_list, synapses_locs

    def save_edges(self):
        """Save edges in dataframe csv."""
        columns = ['pre_partner', 'post_partner', 'weight', 'synapses_locs']
        edge_list = np.array(self.edge_list)
        df = pd.DataFrame(list(zip(edge_list[:, 0], edge_list[:, 1], self.weights,
                          self.synapses_locs)), columns=columns)

        df.to_csv(self.output_edges_path)

        return df

    def create_edges_graph(self):
        """Create edges of the graph."""
        if len(self.weights) == 0:
            self.g.add_edges_from(self.edge_list)
            print("## Info : Edges created!")
        else:
            for i in range(len(self.edge_list)):
                self.g.add_edge(self.edge_list[i][0], self.edge_list[i][1], weight=self.weights[i])

            print("## Info : Edges and weights created!")

    def create_graph(self):
        """Create the graph accessing DB or reading existing file."""
        # access the database and generate graph characteristics if it was not
        # already existing or if it was existing but overwrite option is True
        if self.overwrite or not os.path.exists(self.output_graph_path):
            # assuming that if there is no output_graph there is no edge_list and
            # adjacency matrix saved either

            # access the DB
            self.__connect_DBs()

            if self.input_method == 'user_list':
                self.neurons_list = sorted(list(set(self.input_neurons_list)))
            elif self.input_method == 'all':
                # WARNING : in 'neuron_db_server.py' there is the limit of 10000 neurons
                # so 10000 neurons will be queried
                self.neurons_list = self.neuron_db.find_neuron({})
            elif self.input_method == 'roi':
                # query neurons if roi.contains(Coordinate(soma_loc))
                # TO IMPLEMENT ...
                pass

            self.nodes_attr = self._get_neurons_info_db()
            self.g = self.create_nodes_graph()

            # create useful dictionaries:
            self.neurons_dict_sf = self.create_neurons_dict_sf()
            self.sf_to_neurons = self.create_sf_dict_neurons()

            print("### Info: Running create_edges_list...")
            self.edge_list, self.synapse_list = self.create_edges_list()
            print("### Info: len(edge_list) NOT filtered:", len(self.edge_list))

            # Pre-processing if user specified edges to add:
            if len(self.add_edge_list):
                for e in self.add_edge_list:
                    self.edge_list.append(e)

                self.edge_list = list(set(self.edge_list))
                print("### Info: Added edges specified by the user, len(edge_list) :", len(self.edge_list))

            self.syns_dict = self.create_syns_dict()

            # Pre-processing if user specified synapses location to exclude
            if len(self.exclude_synapses):
                print("### Info: deleting false synapses ...")
                for es in self.exclude_synapses:
                    to_del = dict(filter(lambda elem: elem[1]['syn_loc'] == es, self.syns_dict.items()))

                for k, v in to_del.items():
                    self.syns_dict.pop(k)

            self.weights, self.edge_list, self.synapses_locs = self.compute_weights()
            print(" ### Info: len(weights) (filtered edge_list): ", len(self.edge_list))

            # save outputs: FILE edges
            self.edge_list_df = self.save_edges()
            # add edges in the graph
            self.create_edges_graph()
            # save graph
            nx.write_gpickle(self.g, self.output_graph_path)

        else:
            # load graph, adj and edge list
            self.g = nx.read_gpickle(self.output_graph_path)
            print("### Info: Graph loaded")
            print("Number of nodes: ", self.g.number_of_nodes())
            self.edge_list_df = pd.read_csv(self.output_edges_path, index_col=0)  # with info on the weights and synapses
            # edge list names
            self.edge_list = list(zip(*map(self.edge_list_df.get, ['pre_partner', 'post_partner'])))
            print("### Info: edge_list loaded")

    def preprocess_graph(self):
        """If preprocessed graph is not existing or overwriting is on."""
        if self.overwrite or not os.path.exists(self.output_graph_pp_path):

            # Preprocessing the graph, given specifications in the config file
            pre_proc = len(self.exclude_neurons) or len(self.tags_to_exclude) or len(self.exclude_edges)
            if pre_proc:

                self.g.remove_nodes_from(self.exclude_neurons)
                filtered_nodes = []
                for tte in self.tags_to_exclude:
                    filtered_nodes.extend([n for n, d in self.g.nodes(data=True) if d['cell_type'] == tte[0]
                                          and tte[1] in d['tags']])

                self.g.remove_nodes_from(filtered_nodes)

                print("Num of nodes (filtered): ", self.g.number_of_nodes())

                self.g.remove_edges_from(self.exclude_edges)

            # rename nodes and overwrite the name of the nodes
            # NOTE : finished tag will be added only to interneuorns with no cell type specified,
            # if instead the cell type is specified (eg basket) the rename will be basket_ and it
            # assumes the interneuron finished

            if len(self.rename_rules):
                rename_dict = dict()

                for rule in self.rename_rules:
                    # rule to query the node of interest
                    query = rule[2]
                    if len(query) == 0:
                        # direct rename
                        rename_dict[rule[0]] = rule[1]
                    elif len(query) == 1:
                        # cell type is specifies (example interneuron_ -> basket_)
                        queried_nodes = [n for n, d in self.g.nodes(data=True) if d[list(query.keys())[0]] == list(query.values())[0]]
                    else:
                        # e.g. specified cell type and finished tag
                        nodes_dict = dict()
                        for (n, d) in self.g.nodes(data=True):
                            nodes_dict[n] = 0
                            for k, v in query.items():
                                if d[k] == v:
                                    nodes_dict[n] += 1

                        queried_nodes = list(dict(filter(lambda val: val[1] == 2, nodes_dict.items())).keys())

                    for node in queried_nodes:

                        if node.find(rule[0]) == 0:
                            rename_dict[node] = node.replace(node[:len(rule[0])], rule[1], 1)

                self.g = nx.relabel_nodes(self.g, rename_dict)

            nx.write_gpickle(self.g, self.output_graph_pp_path)

        else:

            self.g = nx.read_gpickle(self.output_graph_pp_path)
            print("Num of nodes (filtered): ", self.g.number_of_nodes())

    def plot_adj_mat(self, configs):  # A, configs, g):
        """
        Plot Adj matrix according to the configs.

        - configs is a dictionary with the following keys:
        ['full', 'some', 'pre', 'post', 'threshold_value'(int/float)];
        - the output paths are the values of the dict;
        configs values will include the list of neuorns of interest.

        If the threshold is present, all the plots will be done considering
        that threshold.
        """
        if configs['adj_plot_thresh'] == 1:
            self.A[self.A <= configs['weights_threshold']] = 0

        self.full_list = list(self.g.nodes())
        fig = plt.figure(figsize=(16, 15))
        ax = fig.add_subplot(111)

        if configs['analysis_type'] == 'adj_plot_all':
            mat = self.A
            ax.set_xticks(np.arange(len(mat)))
            ax.set_xticklabels(self.full_list, rotation=75)
            ax.set_yticks(np.arange(len(mat)))
            ax.set_yticklabels(self.full_list)
        elif configs['analysis_type'] == 'adj_plot_pre':
            mat = self.A[:, [self.full_list.index(i) for i in configs['list']]]
            ax.set_xticks(np.arange(mat.shape[1]))
            ax.set_xticklabels(configs['list'], rotation=75)
            ax.set_yticks(np.arange(mat.shape[0]))
            ax.set_yticklabels(self.full_list)

            # save output file for synapses proofreading
            self.small_list = configs['list']
            self.debug_spec_edges(an_type='pres')

        elif configs['analysis_type'] == 'adj_plot_post':
            mat = self.A[[self.full_list.index(i) for i in configs['list']], :]
            ax.set_xticks(np.arange(mat.shape[1]))
            ax.set_xticklabels(self.full_list, rotation=75)
            ax.set_yticks(np.arange(mat.shape[0]))
            ax.set_yticklabels(configs['list'])

            # save output file for synapses proofreading
            self.small_list = configs['list']
            self.debug_spec_edges(an_type='posts')

        elif configs['analysis_type'] == 'adj_plot_some':
            mat = nx.to_numpy_matrix(self.g, nodelist=configs['list'])
            ax.set_xticks(np.arange(len(configs['list'])))
            ax.set_xticklabels(configs['list'], rotation=75)
            ax.set_yticks(np.arange(len(configs['list'])))
            ax.set_yticklabels(configs['list'])

            # save output file for synapses proofreading
            self.small_list = configs['list']
            self.debug_spec_edges(an_type='some')

        else:
            print("### Info: analysis_type specified not implemented! Exiting...")
            exit()

        i = ax.imshow(mat)
        plt.colorbar(i, ax=ax)
        fig.savefig(self.directory + '/' + configs['output_plot'])

    def debug_spec_edges(self, an_type=[]):
        """Debug edges: proofread output."""
        if self.debug_edges and len(an_type) == 0:
            el_to_save = self.debug_edges_list
        elif an_type == "pres":
            el_to_save = [[pre, post] for pre in self.full_list for post in self.small_list]
            print("### Info: Saving debug edges pre partners...")
        elif an_type == "posts":
            el_to_save = [[pre, post] for pre in self.small_list for post in self.full_list]
            print("### Info: Saving debug edges post partners...")
        elif an_type == "some":
            el_to_save = [[pre, post] for pre in self.small_list for post in self.small_list]
            print("### Info: Saving debug edges for small Adjacency mat...")

        deb_edge_list = pd.DataFrame()
        for i in range(len(el_to_save)):
            q = self.edge_list_df[(self.edge_list_df['pre_partner'] == el_to_save[i][0]) &
                             (self.edge_list_df['post_partner'] == el_to_save[i][1])]
            if len(q) > 0:
                deb_edge_list = deb_edge_list.append(q, ignore_index=True)

        deb_edge_list.to_csv(self.output_debug_edges_path)


if __name__ == '__main__':

    g = ExtractGraph(sys.argv[1])
    # g.build_graph(sys.argv[1])
