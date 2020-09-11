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
# import pickle
import compress_pickle

from .database_synapses import SynapseDatabase
from .database_superfragments import SuperFragmentDatabase
sys.path.insert(0, '/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/repos/funlib.show.neuroglancer')
# sys.path.insert(0, '/n/groups/htem/Segmentation/tmn7/segwaytool.proofreading')
sys.path.insert(0, '/n/groups/htem/Segmentation/shared-dev/cb2_segmentation/segway/synful_tasks')

import segway.dahlia.db_server

logger = logging.getLogger(__name__)


class SynapseGraph():
    """SynapseGraph allows the creation of the graph.

    Also outputs and plots are generated.
    """

    def __init__(self, config_file, overwrite=False):
        """Initialize attributes."""
        self.config_file = config_file
        self.sub_neuron_synapse_ids = {}
        self.predefined_sub_neuron_types = ['axon', 'dendrite', 'soma']
        self.pair_synapses = defaultdict(list)

        self.__initialize_configs(config_file)
        self.__read_configs(config_file)
        self.__check_configs()

        self.overwrite = overwrite
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)

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

        self.csv_output_dir = None
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
        self.rename_rules2 = []
        self.plots = []
        self.weights_with_dist = False
        self.presynapse_exclusion_list = []
        self.postsynapse_exclusion_list = []

        self._named_lists = {}

    def get_pickle_file_path(self):
        return os.path.join(self.graph_dir + '/output_graph.gz')

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

    def get_output_csv_fname(self, basename):
        if self.output_prepend_config_name:
            basename = self.config_name + '_' + basename
        return os.path.join(self.csv_output_dir, basename)

    def __check_configs(self):

        # make sure that essential params are defined
        for p in [
                'db_name', 'db_host', 'db_name_n', 'input_roi_offset',
                'input_roi_size', 'voxel_size_xyz', 'syn_score_threshold',
                'input_method', 'mode_weights',
                ]:
            assert hasattr(self, p), "Paramter %s was not defined in config" % p

        # print(self.debug_edges_list)
        if self.csv_output_dir is None:
            self.csv_output_dir = os.path.join(self.output_dir, 'edge_csv')

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
        logger.debug("Checking existing graph...")
        exist_nodes_list = list(self.g.nodes())
        if not set(exist_nodes_list).issuperset(set(self.neurons_list)):
            new_neurons = [nn for nn in self.neurons_list if nn not in exist_nodes_list]

            # add nodes with attributes
            attr = {}
            logger.debug("Adding new neurons:")
            for nn in new_neurons:
                logger.debug(nn)
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
            new_synapses = self.create_syns_dict(sl)

            # Pre-processing if user specified synapses location to exclude
            # (only new synapses are considered)
            if len(self.exclude_synapses):
                logger.debug("### Info: deleting false synapses ...")
                for es in self.exclude_synapses:
                    to_del = dict(filter(lambda elem: elem[1]['syn_loc'] == es, new_synapses.items()))

                    for k, v in to_del.items():
                        new_synapses.pop(k)

            weights, fil_edge_list, syns_locs = self.compute_weights(new_synapses, el)
            self.edge_list.extend(fil_edge_list)
            tot_weights = list(self.edge_list_df['weight'])
            tot_weights.extend(weights)
            self.weights = tot_weights
            old_syns_locs = [literal_eval(sloc) for sloc in self.edge_list_df.loc[:,'synapses_locs'].values]
            old_syns_locs.extend(syns_locs)
            self.synapses_locs = old_syns_locs
            # save outputs: FILE edges
            self.edge_list_df = self.make_edge_list_df()

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

        neuron_db = segway.dahlia.db_server.NeuronDBServer(
                    db_name=self.db_name_n,
                    host=self.db_host,)
        neuron_db.connect()

        self.syn_db = syn_db
        self.sf_db = sf_db
        self.neuron_db = neuron_db

    def _get_neurons_info_db(self):
        self.neuron_info = {}
        for nid in self.neurons_list:
            node = self.neuron_db.get_neuron(nid)
            node_attr = node.to_json()

            sub_neuron_segs = defaultdict(set)
            for child in node.children:
                child_type = child.split('.')[1]
                found = False
                for subtype in self.predefined_sub_neuron_types:
                    if subtype in child_type:
                        segs = set([int(k) for k in node.segments_by_children[child]])
                        sub_neuron_segs[subtype] |= segs
                        found = True
                if not found:
                    logger.info(f"{child} is not in {self.predefined_sub_neuron_types}")

            node_attr['sub_neuron_segs'] = sub_neuron_segs
            node_attr['presyn_partners'] = set()
            node_attr['postsyn_partners'] = set()

            self.neuron_info[nid] = node_attr
            self.sub_neuron_synapse_ids[nid] = defaultdict(set)

    def _get_neurons_attrs(self):
        nodes_attr = {}
        for nid in self.neurons_list:
            neuron = self.neuron_info[nid]
            nodes_attr[nid] = {}
            nodes_attr[nid]['cell_type'] = neuron['cell_type']
            nodes_attr[nid]['x'] = neuron['soma_loc']['x']
            nodes_attr[nid]['y'] = neuron['soma_loc']['y']
            nodes_attr[nid]['z'] = neuron['soma_loc']['z']
            nodes_attr[nid]['tags'] = neuron['tags']
            nodes_attr[nid]['finished'] = neuron['finished']
        return nodes_attr

    def create_nodes_graph(self):
        """Create nodes of networkx graph."""
        G = nx.DiGraph()
        for i, n in enumerate(self.neurons_list):
            G.add_node(n)

        nx.set_node_attributes(G, self.nodes_attr)
        logger.debug("### Info : Number of nodes in the graph : ", G.number_of_nodes())

        return G

    def create_neurons_dict_sf(self):
        """Dictionary with neurons as keys and their sf as values."""
        neurons_dict_sf = dict()

        for nid in self.neurons_list:
            # superfragments = self.neuron_db.get_neuron(nid).to_json()['segments']
            superfragments = self.neuron_info[nid]['segments']
            sfs_list = [int(k) for k in superfragments]
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
                    self.neuron_info[nid]['postsyn_partners'].add(post_neuron)

            pre_partners_sf = sf['pre_partners']
            # print("pre_partners_sf:", pre_partners_sf)
            for pre_sf in pre_partners_sf:
                if pre_sf not in self.sf_to_neurons:
                    # post neuron not in input list
                    continue
                pre_neuron = self.sf_to_neurons[pre_sf]
                if pre_neuron != nid:
                    edge_list.add((pre_neuron, nid))
                    self.neuron_info[nid]['presyn_partners'].add(pre_neuron)

            for subtype in self.predefined_sub_neuron_types:
                if sf['id'] in self.neuron_info[nid]['sub_neuron_segs'][subtype]:
                    self.sub_neuron_synapse_ids[nid][subtype] |= set(sf['syn_ids'])

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
        logger.info("###: Info: querying synapses DB")
        start = time.time()
        query = {'$and': [{'id': {'$in': list(synapses_list)}},
                {'score': {'$gt': self.syn_score_threshold}}]}

        synapses_query = [syn for syn in self.syn_db.synapses.find(query)]
        logger.info("Synapses query took %f s" % (time.time() - start))

        syns_dict = dict()
        for i, syn in enumerate(synapses_query):
            # take location to store synapses length/distance
            # pre = np.array([syn['pre_x'], syn['pre_y'], syn['pre_z']])
            # post = np.array([syn['post_x'], syn['post_y'], syn['post_z']])
            syn_id = syn['id']
            syns_dict[syn_id] = {'syn_loc': [int(syn['x'] / self.voxel_size_xyz[0]),
                                        int(syn['y'] / self.voxel_size_xyz[1]),
                                        int(syn['z'] / self.voxel_size_xyz[2])],
                            # 'area': syn['area'],
                            # 'dist': np.linalg.norm(pre - post),
                            'sf_pre': syn['id_superfrag_pre'],
                            'sf_post': syn['id_superfrag_post'],
                            'pre_loc': [syn['pre_x'], syn['pre_y'], syn['pre_z']],
                            'post_loc': [syn['post_x'], syn['post_y'], syn['post_z']],
                            }

        logger.info("Synapses dict creation took %f s" % (time.time() - start))

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

        logger.debug("## Info : Computing the weights of the graph...")
        start = time.time()

        syn_weights = defaultdict(float)

        synapses_dict = defaultdict(list)

        for syn_id, syn in syns_dict.items():
            pre_neuron = syn['sf_pre']
            post_neuron = syn['sf_post']
            if pre_neuron not in self.sf_to_neurons or post_neuron not in self.sf_to_neurons:
                continue

            pre_neuron = self.sf_to_neurons[pre_neuron]
            post_neuron = self.sf_to_neurons[post_neuron]

            if pre_neuron == post_neuron:
                continue

            self.pair_synapses[(pre_neuron, post_neuron)].append(syn_id)

            weight = 1
            if mode_area:
                assert False, "To implement/validate"
                weight = syn['area'] / 1e+3
            if mode_length:
                assert False, "To implement/validate"
                # weight = syn['dist']
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
                logger.debug("Edge %s not found in synapse attributes" % str(e))
                filt_edge_list.remove(e)
                # continue
            # assert e in syn_weights
            # weights.append(syn_weights[e])

        logger.debug("Weights creation took %f s" % (time.time() - start))

        return weights, filt_edge_list, synapses_locs

    def make_edge_list_df(self):
        """Save edges in dataframe csv."""
        columns = ['pre_partner', 'post_partner', 'weight', 'synapses_locs']
        edge_list = np.array(self.edge_list)
        df = pd.DataFrame(list(zip(edge_list[:, 0], edge_list[:, 1], self.weights,
                          self.synapses_locs)), columns=columns)
        return df

    def create_edges_graph(self):
        """Create edges of the graph."""
        if len(self.weights) == 0:
            self.g.add_edges_from(self.edge_list)
            logger.debug("## Info : Edges created!")
        else:
            for i in range(len(self.edge_list)):
                self.g.add_edge(self.edge_list[i][0], self.edge_list[i][1], weight=self.weights[i])
            logger.debug("## Info : Edges and weights created!")

    def create_graph(self):
        """Create the graph accessing DB or reading existing file."""
        # access the database and generate graph characteristics if it was not
        # already existing or if it was existing but overwrite option is True

        # access the DB
        self.__connect_DBs()

        if self.input_method == 'user_list':
            self.neurons_list = sorted(list(set(self.input_neurons_list)))
            self.neurons_list = self.add_named_lists(self.neurons_list)
        elif self.input_method == 'all':
            # WARNING : in 'neuron_db_server.py' there is the limit of 10000 neurons
            # so 10000 neurons will be queried
            self.neurons_list = self.neuron_db.find_neuron({})
        elif self.input_method == 'roi':
            # query neurons if roi.contains(Coordinate(soma_loc))
            # TO IMPLEMENT ...
            pass

        if self.overwrite or not os.path.exists(self.get_pickle_file_path()):
            # assuming that if there is no output_graph there is no edge_list and
            # adjacency matrix saved either
            logger.info("Creating graph DB...")
            start_time = time.time()

            self._get_neurons_info_db()
            logger.info(f"Query neuron DB took {time.time() - start_time}s")

            self.nodes_attr = self._get_neurons_attrs()

            self.g = self.create_nodes_graph()
            logger.info(f"create_nodes_graph took {time.time() - start_time}s")

            self.neurons_dict_sf = self.create_neurons_dict_sf()
            self.sf_to_neurons = self.create_sf_dict_neurons()
            logger.info(f"create_neurons_dict_sf took {time.time() - start_time}s")

            self.edge_list, self.synapse_list = self.create_edges_list()
            logger.info(f"create_edges_list took {time.time() - start_time}s")

            # Pre-processing if user specified edges to add:
            if len(self.add_edge_list):
                for e in self.add_edge_list:
                    self.edge_list.append(e)

                self.edge_list = list(set(self.edge_list))
                logger.debug("### Info: Added edges specified by the user, len(edge_list) :", len(self.edge_list))

            self.synapse_info = self.create_syns_dict(self.synapse_list)
            logger.info(f"create_syns_dict took {time.time() - start_time}s")

            # Pre-processing if user specified synapses location to exclude
            if len(self.exclude_synapses):
                count = 0
                for es in self.exclude_synapses:
                    to_del = dict(filter(lambda elem: elem[1]['syn_loc'] == es, self.synapse_info.items()))
                    for k, v in to_del.items():
                        self.synapse_info.pop(k)
                        count += 1
                logger.debug("### Info: deleted %d false synapses..." % count)

            self.weights, self.edge_list, self.synapses_locs = self.compute_weights(
                self.synapse_info, self.edge_list)
            logger.debug(" ### Info: len(weights) (filtered edge_list): ", len(self.edge_list))

            self.edge_list_df = self.make_edge_list_df()
            self.create_edges_graph()

            logger.info("Writing pickle files...")
            compress_pickle.dump((
                self.g,
            # self.neurons_dict_sf,
            self.neuron_info,
            # self.edge_list,
            self.synapse_list,
            self.synapse_info,
            self.sub_neuron_synapse_ids,
            # self.weights,
            # self.synapses_locs,
            self.edge_list_df,
            self.pair_synapses,
                ), self.get_pickle_file_path())

            logger.info(f"Creating graph DB took {time.time() - start_time}")

        else:
            # load graph, adj and edge list
            if False:
                self.g = nx.read_gpickle(self.get_graph_file_path())
                logger.debug("### Info: Graph loaded")
                logger.debug("Number of nodes: ", self.g.number_of_nodes())
                self.edge_list_df = pd.read_csv(self.get_edges_file_path(), index_col=0)  # with info on the weights and synapses
                # edge list names
                self.edge_list = list(zip(*map(self.edge_list_df.get, ['pre_partner', 'post_partner'])))
                logger.debug("### Info: edge_list loaded")

            # with open(self.get_pickle_file_path(), 'rb') as f:
            (
            self.g,
            # self.neurons_dict_sf,
            self.neuron_info,
            # self.edge_list,
            self.synapse_list,
            self.synapse_info,
            self.sub_neuron_synapse_ids,
            # self.weights,
            # self.synapses_locs,
            self.edge_list_df,
            self.pair_synapses,
            ) = compress_pickle.load(self.get_pickle_file_path())

    def preprocess_graph(self):
        """If preprocessed graph is not existing or overwriting is on."""

        # Preprocessing the graph, given specifications in the config file
        pre_proc = len(self.exclude_neurons) or len(self.tags_to_exclude) or len(self.exclude_edges)
        if pre_proc:

            self.g.remove_nodes_from(self.exclude_neurons)
            filtered_nodes = []
            for tte in self.tags_to_exclude:
                filtered_nodes.extend([n for n, d in self.g.nodes(data=True) if d['cell_type'] == tte[0]
                                      and tte[1] in d['tags']])

            self.g.remove_nodes_from(filtered_nodes)

            logger.debug("Num of nodes (filtered): ", self.g.number_of_nodes())

            self.g.remove_edges_from(self.exclude_edges)

        # rename nodes and overwrite the name of the nodes
        # NOTE : finished tag will be added only to interneuorns with no cell type specified,
        # if instead the cell type is specified (eg basket) the rename will be basket_ and it
        # assumes the interneuron finished

        self.preprocess_full_list = list(self.g.nodes())

        self.rename_dict = dict()
        if len(self.rename_rules):
            for rule in self.rename_rules:
                # rule to query the node of interest
                from_name = rule[0]
                to_name = rule[1]
                query = {}
                if len(rule) >= 3:
                    query = rule[2]

                if len(query) == 0:
                    # direct rename
                    self.rename_dict[from_name] = to_name
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

                    queried_nodes = list(
                        dict(filter(lambda val: val[1] == 2, nodes_dict.items())).keys())

                for node in queried_nodes:

                    if node.find(from_name) == 0:
                        replace = node.replace(node[:len(from_name)], to_name, 1)
                        self.rename_dict[node] = replace

        if len(self.rename_rules2):
            for rule in self.rename_rules2:
                criterias = rule[0]
                actions = rule[1]

                neuron_list = self.get_filtered_list(criterias)

                for neuron in neuron_list:
                    current_name = self.rename_dict.get(neuron, neuron)
                    new_name = self.apply_rename_action(current_name, actions)
                    self.rename_dict[neuron] = new_name

        self.rename_dict_reverse = dict()
        for k in self.rename_dict:
            self.rename_dict_reverse[self.rename_dict[k]] = k

        # if len(self.rename_dict):

        #     # self.g = nx.relabel_nodes(self.g, self.rename_dict)

        #     for l in [self.presynapse_exclusion_list, self.postsynapse_exclusion_list, self.neurons_list]:
        #         for i, n in enumerate(l):
        #             if n in self.rename_dict:
        #                 l[i] = self.rename_dict[n]

    # def save_user_edges_debug(self):
    #     if self.debug_edges_list is not None and \
    #             len(self.debug_edges_list) == 2:
    #         self.save_edges_to_csv(self.debug_edges_list[0], self.debug_edges_list[1])

    def apply_rename_action(self, name, actions):

        for action in actions:
            action_type = action[0]
            if action_type == "replace":
                name = name.replace(action[1], action[2])
            elif action_type == "prepend":
                name = action[1] + name
            else:
                assert False, "Action %s not supported yet!" % action_type

        return name

    def check_criterias(self, name, criterias):
        neuron = self.g.node[name]
        for c in criterias:
            if c == "soma_x_min":
                if neuron['x'] <= criterias[c]:
                    return False
            elif c == "soma_x_min_div16":
                if neuron['x']/16 <= criterias[c]:
                    return False
            elif c == "soma_x_max":
                if neuron['x'] > criterias[c]:
                    return False
            elif c == "soma_x_max_div16":
                if neuron['x']/16 > criterias[c]:
                    return False
            elif c == "soma_y_min":
                if neuron['y'] <= criterias[c]:
                    return False
            elif c == "soma_y_min_div16":
                if neuron['y']/16 <= criterias[c]:
                    return False
            elif c == "soma_y_max":
                if neuron['y'] > criterias[c]:
                    return False
            elif c == "soma_y_max_div16":
                if neuron['y']/16 > criterias[c]:
                    return False
        return True

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

        # el_to_save = [pre_list, post_list]
        el_to_save = [[pre, post] for pre in pre_list for post in post_list]

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
        deb_edge_list.to_csv(fname + '.csv')

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

    def add_named_lists(self, l):
        ll = []
        # print(l)
        for n in l:

            # print(n)

            if isinstance(n, dict):
                n = self.get_filtered_list(n)
                for nn in n:
                    ll.append(nn)

            elif n in self._named_lists:
                nl = self._named_lists[n]
                for nn in nl:
                    ll.append(nn)

            else:
                ll.append(n)
        return ll

    def expand_list(self, neuron_list):
        return self.add_named_lists(neuron_list)

    def rename_list(self, neuron_list, reverse=False):

        dictionary = self.rename_dict if not reverse else self.rename_dict_reverse

        # ll = self.expand_list(neuron_list)

        ret = []
        for n in neuron_list:
            if n in dictionary:
                n = dictionary[n]
            ret.append(n)

        return ret

    def get_filtered_list(self, criterias):

        if 'list' in criterias:
            unfiltered_list = criterias['list']
            criterias.pop('list')
        else:
            unfiltered_list = self.preprocess_full_list
        unfiltered_list = self.add_named_lists(unfiltered_list)

        filtered_list = []
        if len(criterias):
            for neuron in unfiltered_list:
                if self.check_criterias(neuron, criterias):
                    filtered_list.append(neuron)
        else:
            filtered_list = unfiltered_list

        return filtered_list

    def get_partners(
            self,
            neuron,
            synapse_type,
            neuron_subtype=None,
            partner_type=None,
            partner_subtype=None,
            condition_fn=None,
            synapse_min_count=0,
            return_synapse_locs=False,
            filter_list=None,
            ):

        assert neuron in self.neuron_info
        assert neuron_subtype is None or neuron_subtype in self.predefined_sub_neuron_types
        assert partner_subtype is None or partner_subtype in self.predefined_sub_neuron_types

        sub_neuron_syns = None
        if neuron_subtype:
            sub_neuron_syns = self.sub_neuron_synapse_ids[neuron][neuron_subtype]

        if synapse_type == 'presyn':
            all_partners = self.neuron_info[neuron]['presyn_partners']
        else:
            all_partners = self.neuron_info[neuron]['postsyn_partners']

        # return all_partners
        if filter_list:
            all_partners = [k for k in all_partners if k in filter_list]

        partners = set()
        syn_locs = {}

        for p in all_partners:

            if synapse_type == 'presyn':
                pair = (p, neuron)
            else:
                pair = (neuron, p)

            if partner_type and self.neuron_info[p]['cell_type'] != partner_type:
                continue

            sub_partner_syns = None
            if partner_subtype:
                sub_partner_syns = self.sub_neuron_synapse_ids[p][partner_subtype]

            all_syn_ids = self.pair_synapses[pair]

            # if neuron == 'interneuron_89':
            #     print(self.sub_neuron_synapse_ids['interneuron_89'])
            #     print(graph.sub_neuron_synapse_ids['interneuron_89'])

            syn_ids = []
            for s in all_syn_ids:
                if neuron_subtype and s not in sub_neuron_syns:
                    continue
                if partner_subtype and s not in sub_partner_syns:
                    continue
                syn_ids.append(s)

            # if len(sub_partner_syns):
            #     # print(f'partner_subtype: {partner_subtype}')
            #     print(f'sub_partner_syns: {sub_partner_syns}')
            #     print(f'all_syn_ids: {all_syn_ids}')
            #     print(f'syn_ids: {syn_ids}')

            if len(syn_ids) > synapse_min_count:
                partners.add(p)
                syn_locs[pair] = []
                for s in syn_ids:
                    syn_locs[pair].append(self.synapse_info[s]['syn_loc'])

        if return_synapse_locs:
            # return (partners, syn_locs)
            return syn_locs
        else:
            return partners



