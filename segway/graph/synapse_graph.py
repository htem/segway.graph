import numpy as np
import networkx as nx
import logging
import os
import sys
import json
import time
from io import StringIO
from collections import defaultdict
from ast import literal_eval
import pandas as pd
from jsmin import jsmin

from database_synapses import SynapseDatabase
from database_superfragments import SuperFragmentDatabase
sys.path.insert(0, '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/repos/funlib.show.neuroglancer')
sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/segwaytool.proofreading')
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway/synful_tasks')
import segwaytool.proofreading
import segwaytool.proofreading.neuron_db_server

logger = logging.getLogger(__name__)


class SynapseGraph():
    """SynapseGraph allows the creation of the graph.

    Also outputs and plots are generated.
    """

    def __init__(self, config_file, overwrite=False):
        """Initialize attributes."""
        self.config_file = config_file

        self.__initialize_configs(config_file)
        self.__read_configs(config_file)
        self.__check_configs()

        self.overwrite = overwrite

        os.makedirs(self.output_dir, exist_ok=True)

        self.__connect_DBs()
        self.create_graph()
        self.__check_existing_graph()
        self.preprocess_graph()

        self.A = nx.to_numpy_matrix(self.g)

    def __initialize_configs(self, input_file):
        """Initialize default values."""
        # create output directory with same name of config file

        output_dir, config_name = os.path.split(self.config_file)
        if output_dir == '':
            output_dir = '.'
        config_name = config_name.split('.')[0]
        output_dir = os.path.join(output_dir, config_name)

        self.output_dir = output_dir
        self.config_name = config_name
        self.output_prepend_config_name = False
        self.graph_dir = output_dir
        # self.overwrite = False
        self.add_edge_list = []
        self.exclude_neurons = []
        self.tags_to_exclude = []
        self.exclude_edges = []
        self.exclude_synapses = []
        self.debug_edges = False
        self.debug_edges_list = None
        self.rename_rules = []
        self.plots = []
        self.weights_with_dist = False
        self.presynapse_exclusion_list = []
        self.postsynapse_exclusion_list = []

        self._named_lists = {}

    def get_graph_file_path(self):
        return os.path.join(self.graph_dir + '/output_graph.gpickle')

    def get_edges_file_path(self):
        return os.path.join(self.graph_dir + '/output_edges.csv')

    def get_debug_edges_file_path(self):
        return os.path.join(self.graph_dir + '/output_debug_edges.csv')

    def get_output_fname(self, basename):
        if self.output_prepend_config_name:
            basename = self.config_name + '_' + basename
        return os.path.join(self.output_dir, basename)

    def __check_configs(self):

        # make sure that essential params are defined
        for p in [
                'db_name', 'db_host', 'db_name_n', 'input_roi_offset',
                'input_roi_size', 'voxel_size_xyz', 'syn_score_threshold',
                'input_method', 'mode_weights',
                ]:
            assert hasattr(self, p), "Paramter %s was not defined in config" % p

        # print(self.debug_edges_list)

    def __read_configs(self, input_file):
        """Recursively read configs from given JSON file."""
        logger.info("Parsing configuration from %s" % input_file)
        with open(input_file) as js_file:
            minified = jsmin(js_file.read())
            params = json.load(StringIO(minified))

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
                # setattr(self, key, getattr(self, key).extend(params[key]))
                getattr(self, key).extend(params[key])
            else:
                # initialize or overwrite value
                setattr(self, key, params[key])

        if 'named_lists' in params:
            for l in params['named_lists']:
                self._named_lists[l] = params['named_lists'][l]

    def __check_existing_graph(self):
        """Check existing graph.

        If it was already stored and update if
        the user-specified neurons list is different from the existing one.
        """
        logger.info("Checking existing graph...")
        exist_nodes_list = list(self.g.nodes())
        if not set(exist_nodes_list).issuperset(set(self.neurons_list)):
            new_neurons = [nn for nn in self.neurons_list if nn not in exist_nodes_list]

            # add nodes with attributes
            attr = {}
            print("Adding new neurons:")
            for nn in new_neurons:
                print(nn)
                neuron = self.neuron_db.get_neuron(nn).to_json()
                # create dictionary with attributes per neuron
                attr = dict()
                attr['cell_type'] = neuron['cell_type']
                attr['x'] = neuron['soma_loc']['x']
                attr['y'] = neuron['soma_loc']['y']
                attr['z'] = neuron['soma_loc']['z']
                attr['tags'] = neuron['tags']
                attr['finished'] = neuron['finished']

                self.g.add_node(nn, attr_dict=attr)

            # update useful dictionary with superfragments
            self.neurons_dict_sf = self.create_neurons_dict_sf()
            self.sf_to_neurons = self.create_sf_dict_neurons()

            for nn in new_neurons:
                el, sl = self.directed_edges_sf(nn)

            el = list(el)
            # query only new synapses
            new_syn_dict = self.create_syns_dict(sl)

            # Pre-processing if user specified synapses location to exclude
            # (only new synapses are considered)
            if len(self.exclude_synapses):
                print("### Info: deleting false synapses ...")
                for es in self.exclude_synapses:
                    to_del = dict(filter(lambda elem: elem[1]['syn_loc'] == es, new_syn_dict.items()))

                    for k, v in to_del.items():
                        new_syn_dict.pop(k)

            weights, fil_edge_list, syns_locs = self.compute_weights(new_syn_dict, el)
            self.edge_list.extend(fil_edge_list)
            tot_weights = list(self.edge_list_df['weight'])
            tot_weights.extend(weights)
            self.weights = tot_weights
            old_syns_locs = [literal_eval(sloc) for sloc in self.edge_list_df.loc[:,'synapses_locs'].values]
            old_syns_locs.extend(syns_locs)
            self.synapses_locs = old_syns_locs
            # save outputs: FILE edges
            self.edge_list_df = self.save_edges()

            # add edges in the graph
            for i in range(len(fil_edge_list)):
                self.g.add_edge(fil_edge_list[i][0], fil_edge_list[i][1], weight=weights[i])

            # save graph
            nx.write_gpickle(self.g, self.get_graph_file_path())

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
        print(self.neurons_list)
        for nid in self.neurons_list:
            print(nid)
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

    def directed_edges_sf(self, nid, edge_list=set(), synapse_list=set()):
        """Given a specific neuron id nid, it looks for all possible edges."""
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

        return edge_list, synapse_list

    def create_edges_list(self):
        """Create edges and synapses lsit."""
        neuron_list = np.array(list(self.neurons_dict_sf.keys()))  # all neurons

        for nid in neuron_list:
            # for each neuron, we get their post partners as sf
            # convert them to neuron_id and add directed edge
            edge_list, synapse_list = self.directed_edges_sf(nid)

        edge_list = list(edge_list)

        return edge_list, synapse_list

    def create_syns_dict(self, synapses_list):
        """Given the list of synapses : a dictionary is created as query result."""
        # get synapse attributes
        print("###: Info: querying synapses DB")
        start = time.time()
        query = {'$and': [{'id': {'$in': list(synapses_list)}},
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

    def compute_weights(self, syns_dict, edge_list):
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

        for k, syn in syns_dict.items():
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
        filt_edge_list = edge_list.copy()

        for e in edge_list:
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

        df.to_csv(self.get_edges_file_path())

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

        if self.overwrite or not os.path.exists(self.get_graph_file_path()):
            # assuming that if there is no output_graph there is no edge_list and
            # adjacency matrix saved either

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

            self.syns_dict = self.create_syns_dict(self.synapse_list)

            # Pre-processing if user specified synapses location to exclude
            if len(self.exclude_synapses):
                count = 0
                for es in self.exclude_synapses:
                    to_del = dict(filter(lambda elem: elem[1]['syn_loc'] == es, self.syns_dict.items()))

                    for k, v in to_del.items():
                        self.syns_dict.pop(k)
                        count += 1

                print("### Info: deleted %d false synapses..." % count)

            self.weights, self.edge_list, self.synapses_locs = self.compute_weights(self.syns_dict,
                                                                                    self.edge_list)
            print(" ### Info: len(weights) (filtered edge_list): ", len(self.edge_list))

            # save outputs: FILE edges
            self.edge_list_df = self.save_edges()
            # add edges in the graph
            self.create_edges_graph()
            # save graph
            nx.write_gpickle(self.g, self.get_graph_file_path())

        else:
            # load graph, adj and edge list
            self.g = nx.read_gpickle(self.get_graph_file_path())
            print("### Info: Graph loaded")
            print("Number of nodes: ", self.g.number_of_nodes())
            self.edge_list_df = pd.read_csv(self.get_edges_file_path(), index_col=0)  # with info on the weights and synapses
            # edge list names
            self.edge_list = list(zip(*map(self.edge_list_df.get, ['pre_partner', 'post_partner'])))
            print("### Info: edge_list loaded")

    def preprocess_graph(self):
        """If preprocessed graph is not existing or overwriting is on."""
        # if self.overwrite or not os.path.exists(self.output_graph_pp_path):

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

        self.rename_dict = dict()
        self.rename_dict_reverse = dict()
        if len(self.rename_rules):
            self.rename_dict = dict()

            for rule in self.rename_rules:
                # rule to query the node of interest
                query = rule[2]
                if len(query) == 0:
                    # direct rename
                    self.rename_dict[rule[0]] = rule[1]
                    self.rename_dict_reverse[rule[1]] = rule[0]
                    continue

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
                        replace = node.replace(node[:len(rule[0])], rule[1], 1)
                        self.rename_dict[node] = replace
                        self.rename_dict_reverse[replace] = node

            # # rename exclusion lists
            # renamed_neuron_list = []
            # neurons_set = set(self.neurons_list)
            # for n in list(self.g.nodes()):
            #     print(n)
            #     if n in neurons_set:
            #         renamed_neuron_list.append(n)
            # self.neurons_list = renamed_neuron_list

            self.g = nx.relabel_nodes(self.g, self.rename_dict)

            for l in [self.presynapse_exclusion_list, self.postsynapse_exclusion_list, self.neurons_list]:
                for i, n in enumerate(l):
                    if n in self.rename_dict:
                        l[i] = self.rename_dict[n]

        # nx.write_gpickle(self.g, self.output_graph_pp_path)

        # else:

        #     self.g = nx.read_gpickle(self.output_graph_pp_path)
        #     print("Num of nodes (filtered): ", self.g.number_of_nodes())

    # def save_user_edges_debug(self):
    #     if self.debug_edges_list is not None and \
    #             len(self.debug_edges_list) == 2:
    #         self.save_edges_to_csv(self.debug_edges_list[0], self.debug_edges_list[1])

    def save_edges_to_csv(self, pre_list=None, post_list=None, fname=None):
        """Debug edges: proofread output."""
        full_list = list(self.g.nodes())

        if pre_list is None:
            pre_list = full_list
        else:
            pre_list = self.rename_list(pre_list, reverse=True)

        if post_list is None:
            post_list = full_list
        else:
            post_list = self.rename_list(post_list, reverse=True)

        # print("full_list:", full_list)
        # print("pre_list:", pre_list)
        # print("post_list:", post_list)
        # print("self.edge_list_df:", self.edge_list_df)

        # el_to_save = [pre_list, post_list]
        el_to_save = [[pre, post] for pre in pre_list for post in post_list]

        # print(self.edge_list_df[
        #         (self.edge_list_df['pre_partner'] == 'purkinje_1') &
        #         (self.edge_list_df['post_partner'] == 'interneuron_87')])
        # exit()

        deb_edge_list = pd.DataFrame()
        for i in range(len(el_to_save)):
            # print("el_to_save[i][0]:", el_to_save[i][0])
            # print("el_to_save[i][1]:", el_to_save[i][1])
            q = self.edge_list_df[
                (self.edge_list_df['pre_partner'] == el_to_save[i][0]) &
                (self.edge_list_df['post_partner'] == el_to_save[i][1])
                ]
            # print(q)
            if len(q) > 0:
                deb_edge_list = deb_edge_list.append(q, ignore_index=True)

        # print("deb_edge_list:", deb_edge_list)
        # exit()
        if fname is None:
            fname = self.get_debug_edges_file_path()
        deb_edge_list.to_csv(fname)

    def get_matrix(self):
        return self.A

    def get_graph(self):
        return self.g

    def get_presynapse_exclusion_list(self):
        return self.presynapse_exclusion_list

    def get_postsynapse_exclusion_list(self):
        return self.postsynapse_exclusion_list

    def get_neurons_list(self):

        # # rename exclusion lists
        # renamed_neuron_list = []
        # neurons_set = set(self.neurons_list)
        # for n in list(self.g.nodes()):
        #     print(n)
        #     if n in neurons_set:
        #         renamed_neuron_list.append(n)
        # self.neurons_list = renamed_neuron_list

        # all_nodes = self.g.nodes()
        # filtered = []
        # for n in all_nodes
        # return self.neurons_list

        return list(self.g.nodes())

    def rename_list(self, l, reverse=False):

        dictionary = self.rename_dict if not reverse else self.rename_dict_reverse

        ll = []
        for n in l:
            if n in self._named_lists:
                nl = self._named_lists[n]
                for nn in nl:
                    ll.append(nn)
            else:
                ll.append(n)

        ret = []
        for n in ll:
            if n in dictionary:
                n = dictionary[n]
            ret.append(n)

        return ret
