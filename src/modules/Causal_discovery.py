from pathlib import Path
import dill


class CausalGraph:
    def __init__(self, data_all, data_train, num_diagnosis, num_procedure, num_medication, dataset):
        root = Path(__file__).absolute().parents[2] / 'data' / dataset / 'graphs'
        def read(name):
            with open(root / name, 'rb') as stream:
                return dill.load(stream)
        self.causal_graph = read('causal_graph.pkl')
        self.diag_med_effect = read('Diag_Med_causal_effect.pkl')
        self.proc_med_effect = read('Proc_Med_causal_effect.pkl')

    def get_graph(self, graph_id, graph_type):
        return self.causal_graph[graph_id][{'Diag': 0, 'Proc': 1, 'Med': 2}[graph_type]]

    def get_effect(self, a, b, A_type, B_type):
        effects = self.diag_med_effect if A_type == 'Diag' else self.proc_med_effect
        return effects.loc[f'{A_type}_{int(a)}', f'{B_type}_{int(b)}']

    def get_threshold_effect(self, threshold, A_type, B_type):
        effects = self.diag_med_effect if A_type == 'Diag' else self.proc_med_effect
        return effects.stack().quantile(threshold)
