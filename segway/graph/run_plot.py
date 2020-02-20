import sys
import json
from jsmin import jsmin
from io import StringIO

from segway.graph.synapse_graph import SynapseGraph
from segway.graph.plot_adj_mat import plot_adj_mat


if __name__ == '__main__':

    config_f = sys.argv[1]
    with open(config_f) as js_file:
        minified = jsmin(js_file.read())
        config = json.load(StringIO(minified))

    overwrite = False
    if len(sys.argv) == 3 and sys.argv[2] == "--overwrite":
        overwrite = True

    g = SynapseGraph(config_f, overwrite=overwrite)

    if 'plots' in config:
        # print(config['plots'])
        for plot_config in config['plots']:
            print("Plotting", plot_config)
            plot_type = config.get('analysis_type', 'adj_plot')
            if plot_type == 'adj_plot':
                plot_adj_mat(g, plot_config)
            else:
                raise RuntimeError("Analysis %s is not implemented" % plot_type)
