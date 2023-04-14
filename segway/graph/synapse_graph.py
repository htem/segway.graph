import logging
import os
import sys
import time
from collections import defaultdict, namedtuple
import itertools
import gzip
import pickle

import numpy as np
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_graph(G):
    for n in G.nodes:
        del G.nodes[n]['superfragments']

def save_nx_graph(G, networkx_path):
    with gzip.open(networkx_path, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

def load_nx_graph(networkx_path, clean=False):
    with gzip.open(networkx_path, 'rb') as f:
        G = pickle.load(f)
    if clean:
        clean_graph(G)
    return G


class SynapseGraph():

    def __init__(
            self,
            neuron_db,
            superfragment_db,
            synapse_db,
            graph_in=None,
    ):
        """
            Args:
                synapse_score_threshold: float | Callable[[syn, neuron1, neuron2, graph], bool]
                    If a float, a synapse score must be above threshold to be included.
                    If a callable, the function is given neuron1 and neuron2 and returns `True` to include the synapse connection, where syn is the synapse node, and neuron1 and neuron2 are the two neurons being connected.
                    By default, the script checks for the `score` field in the synapse database for the score. If this field is not available, add the connection anyway.
        """

        self.neuron_db = neuron_db
        self.superfragment_db = superfragment_db
        self.synapse_db = synapse_db

        self.predefined_sub_neuron_types = ['axon', 'dendrite', 'soma']

        self.sfid_to_nid = {}
        self.graph_in = graph_in
        self.G = nx.MultiDiGraph()

        if graph_in is not None:
            if isinstance(graph_in, str):
                # graph_in is a path
                if os.path.exists(graph_in):
                    self.G = load_nx_graph(graph_in)
            else:
                assert issubclass(graph_in, nx.Graph)
                self.G = graph_in

    def get_graph(self):
        return self.G

    def save(self, fout=None):
        if fout is None and isinstance(graph_in, str):
            fout = graph_in
        save_nx_graph(self.G, fout)

    @staticmethod
    def get_base_name(nid):
        return str(nid).split('.')[0]

    def add_neurons(self,
                    neuron_list,
                    neuron_list_all_from=None,
                    neuron_list_all_to=None,
                    add_node_attributes=None,
                    add_edge_attributes=None,
                    synapse_score_fn=None,
                    synapse_score_threshold=None,
                    include_autapses=False,
                    ):

        if neuron_list_all_from is None:
            neuron_list_all_from = []
        if neuron_list_all_to is None:
            neuron_list_all_to = []

        if type(synapse_score_threshold) in [int, float]:
            assert synapse_score_fn is not None

        assert isinstance(neuron_list, list)
        assert isinstance(neuron_list_all_from, list)
        assert isinstance(neuron_list_all_to, list)

        neuron_list = list(set(neuron_list) | set(neuron_list_all_from) |
                           set(neuron_list_all_to))

        node_data, subnode_data = self._get_node_data(neuron_list)

        def unchanged(node):
            if node.name in self.G.nodes:
                if (set(node.data['segments']) ==
                    set(self.G.nodes[node.name]['superfragments'])):
                    return True
            return False

        # filter out unchanged objects existed in the current graph
        node_data = list(itertools.filterfalse(unchanged, node_data))

        self._add_nodes(node_data, add_node_attributes)

        # construct superfragment_id -> neuron_id mapping
        self.sfid_to_nid = {}
        for node in self.G.nodes:
            for sfid in self.G.nodes[node]['superfragments']:
                self.sfid_to_nid[sfid] = node

        synapse_data = self._get_synapse_data(node_data)

        self._add_edges(synapse_data=synapse_data,
                        neuron_list_all_from=neuron_list_all_from,
                        neuron_list_all_to=neuron_list_all_to,
                        synapse_score_threshold=synapse_score_threshold,
                        add_edge_attributes=add_edge_attributes,
                        include_autapses=include_autapses,
                        synapse_score_fn=synapse_score_fn,
                        )

        # TODO: get neuron names of unmapped superfragments if available
        # self._add_names_to_found_superfragments()

    def _add_names_to_found_superfragments(self):
        nodes = list(self.G.nodes)
        for sid in nodes:
            if not sid.isdigit():
                continue
            nid = self.neuron_db.find_neuron_with_segment_id(int(sid))
            if nid is None:
                continue
            nid = self.get_base_name(nid)
            self.G.add_node(nid)
            self.G = nx.contracted_nodes(self.G, nid, sid, copy=False)  # merge sid to nid

    def _get_node_data(self, neuron_list):
        NodeInfo = namedtuple('NodeInfo', ['name', 'data'])
        ret = []
        ret_subnode = []
        for nid in neuron_list:
            data = self.neuron_db.get_neuron(nid)
            data_ = data.to_json()
            data_['segments'] = [int(k) for k in data_['segments']]
            ret.append(NodeInfo(nid, data_))

            for child in data.children:
                child_data = self.neuron_db.get_neuron(child).to_json()
                child_data['segments'] = [int(k) for k in child_data['segments']]
                ret_subnode.append(NodeInfo(child, child_data))

        return ret, ret_subnode

    def _add_nodes(self, nodes, add_node_attributes=None):
        """Add nodes with data to G. Also remove superfragments that were added as nodes
        but are part of the new nodes.

        We will not add unidentified superfragments here but later as needed.
        """

        # first remove duplicated nodes
        self.G.remove_nodes_from([k.name for k in nodes])

        # remove nodes that might be unmapped superfragments
        for node in nodes:
            self.G.remove_nodes_from([str(k) for k in node.data['segments']])

        # add nodes
        for node in nodes:
            attrs = {'superfragments': node.data['segments']}
            if add_node_attributes is not None:
                attrs.update(add_node_attributes(node))
            self.G.add_node(node.name, **attrs)

    def _get_synapse_data(self, nodes):

        # first get the list of syns using superfragments' mapped synapses
        syn_ids = []
        for node in nodes:
            superfragments = node.data['segments']
            sf_infos = self.superfragment_db.get_list(superfragments)
            for sf in sf_infos:
                syn_ids.extend(sf['syn_ids'])

        # get syn data
        syn_ids = list(set(syn_ids))
        ret = []
        for syn_id in syn_ids:
            syn_info = self.synapse_db.get(syn_id)
            ret.append(syn_info)
        return ret

    def _add_edges(self, synapse_data,
                   neuron_list_all_from=None,
                   neuron_list_all_to=None,
                   synapse_score_threshold=None,
                   add_edge_attributes=None,
                   include_autapses=False,
                   synapse_score_fn=None,
                   ):
        """
            Args:
                add_edge_attributes: Callable[syn, dict[str, Any]]
                    Returns a dict of attributes to be added to the edge (synapse).

                synapse_score_fn: Callable[syn, float]
                    Returns the score of the synapse.
        """
        for syn in synapse_data:
            presyn_nid = self.sfid_to_nid.get(syn['id_superfrag_pre'])
            postsyn_nid = self.sfid_to_nid.get(syn['id_superfrag_post'])

            # by default we only keep connections where both pre and post superfragments are within the node list, but `neuron_list_all_from` and `neuron_list_all_to` override this behavior
            if presyn_nid is None:
                assert postsyn_nid is not None
                if postsyn_nid not in neuron_list_all_to:
                    continue
                presyn_nid = self.neuron_db.find_neuron_with_segment_id(
                                                        syn['id_superfrag_pre'])
                if presyn_nid is None:
                    presyn_nid = syn['id_superfrag_pre']
            if postsyn_nid is None:
                assert presyn_nid is not None
                if presyn_nid not in neuron_list_all_from:
                    continue
                postsyn_nid = self.neuron_db.find_neuron_with_segment_id(
                                                        syn['id_superfrag_post'])
                if postsyn_nid is None:
                    postsyn_nid = syn['id_superfrag_post']

            # by default we skip autapses which are usually false positive predictions
            if not include_autapses:
                if self.get_base_name(presyn_nid) == self.get_base_name(postsyn_nid):
                    continue

            # skip syn if synapse_score_threshold is not matched
            if synapse_score_threshold is not None:
                if type(synapse_score_threshold) in [float, int]:
                    if synapse_score_fn(syn) < synapse_score_threshold:
                        continue
                elif synapse_score_threshold(syn, presyn_nid, postsyn_nid, self.G) == False:
                    continue

            attrs = {}
            if add_edge_attributes is not None:
                attrs.update(add_edge_attributes(syn))

            # add unmapped superfragments if any
            for nid in (presyn_nid, postsyn_nid):
                if type(nid) == int:
                    self.G.add_node(str(nid))

            # finally add the edge
            logger.debug(f'{presyn_nid} to {postsyn_nid}: {attrs}')
            self.G.add_edge(self.get_base_name(presyn_nid), self.get_base_name(postsyn_nid),
                            **attrs)

