import sys
import json

from segway.graph.synapse_graph import SynapseGraph
from segway.graph.plot_adj_mat import plot_adj_mat


if __name__ == '__main__':

    config_f = sys.argv[1]
    with open(config_f) as f:
        config = json.load(f)

    overwrite = False
    if len(sys.argv) == 3 and sys.argv[2] == "--overwrite":
        overwrite = True

    g = SynapseGraph(config_f, overwrite=overwrite)

    if 'debug_edges' in config and config['debug_edges']:
        g.save_user_edges_debug()  # debug specified edges

    if 'plots' in config:
        # print(config['plots'])
        for plot_config in config['plots']:
            print("Plotting", plot_config)
            plot_adj_mat(g, plot_config)
