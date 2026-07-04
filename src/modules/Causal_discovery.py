import os
import dill
import networkx as nx
import statsmodels.api as sm
import pandas as pd
from cdt.causality.graph import GES,GIES
from dowhy import CausalModel
from tqdm import tqdm

class CausalGraph:
    def __init__(self,data_all, data_train, num_diagnosis, num_procedure, num_medication, dataset):
        self.dataset = dataset

        self.num_d = num_diagnosis
        self.num_p = num_procedure
        self.num_m = num_medication

        self.data = self.data_process(data_train)

        # d-d p-p m-m
        self.causal_graph = self.build_graph(data_all)

        # causal effect
        self.diag_med_effect = self.build_effect(num_diagnosis, num_medication, "Diag", "Med")
        self.proc_med_effect = self.build_effect(num_procedure, num_medication, "Proc", "Med")

    def get_graph(self,graph_id,graph_type):
        graph = self.causal_graph[graph_id]

        if graph_type == "Diag":
            return graph[0]
        elif graph_type == "Proc":
            return graph[1]
        elif graph_type == "Med":
            return graph[2]

    def get_effect(self, a, b, A_type, B_type):
        a = A_type + '_' + str(int(a))
        b = B_type + '_' + str(int(b))

        if A_type == "Diag" and B_type == "Med":
            effect_df = self.diag_med_effect
        elif A_type == "Proc" and B_type == "Med":
            effect_df = self.proc_med_effect
        else:
            raise ValueError("Invalid A_type and B_type combination")

        effect = effect_df.loc[a, b]
        return effect

    def get_threshold_effect(self, threshold, A_type, B_type):
        if A_type == "Diag" and B_type == "Med":
            effect_df = self.diag_med_effect
        elif A_type == "Proc" and B_type == "Med":
            effect_df = self.proc_med_effect
        else:
            raise ValueError("Invalid combination")

        flattened = effect_df.stack()

        # compute threshold value
        threshold_value = flattened.quantile(threshold)
        return threshold_value

    def build_effect(self, num_a, num_b, a_type, b_type):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, f"../../data/{self.dataset}/graphs/{a_type}_{b_type}_causal_effect.pkl")
        # 构建完整的文件路径
        try:
            effect_df = dill.load(open(file_path, "rb"))
        except FileNotFoundError:
            print(f"你的本地没有关于的基于无图的因果效应，正在建立中，这大概需要几个小时的时间..")

            processed_data = self.data
            effect_df = pd.DataFrame(0.0, index=[f"{a_type}_{i}" for i in range(num_a)],
                                     columns=[f"{b_type}_{j}" for j in range(num_b)])

            for i in tqdm(range(num_a)):
                for j in range(num_b):
                    causal_value = self.compute_causal_value(processed_data, i, j, a_type, b_type)
                    effect_df.at[f"{a_type}_{i}", f"{b_type}_{j}"] = causal_value
                    print(f"{a_type}:{i}, {b_type}:{j}, causal_value:{causal_value}")

            with open(file_path, "wb") as f:
                dill.dump(effect_df, f)

        return effect_df

    def compute_causal_value(self, data, d, m, a_type, b_type):
        selected_data = data[[f'{a_type}_{d}', f'{b_type}_{m}']]
        model = CausalModel(data=selected_data, treatment=f'{a_type}_{d}', outcome=f'{b_type}_{m}')
        identified_estimate = model.identify_effect(proceed_when_unidentifiable=True)
        # backdoor
        estimate = model.estimate_effect(identified_estimate,
                                         method_name="backdoor.generalized_linear_model",
                                         #method_params={"matching_algorithm": "nearest", "distance_metric": "mahalanobis","glm_family": sm.families.Binomial()})
                                         method_params={"glm_family": sm.families.Binomial()})
        return estimate.value


    def build_graph(self, data_all):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Build full file path
        file_path = os.path.join(current_dir, f"../../data/{self.dataset}/graphs/causal_graph.pkl")
        try:
            subgraph_list = dill.load(open(file_path, "rb"))
        except FileNotFoundError:
            causal_graphs = []
            print("Build all causal graph")
            sessions = self.sessions_process(data_all)
            for adm in tqdm(sessions):
                D = adm[0]
                P = adm[1]
                M = adm[2]

                visit = [f"Diag_{d}" for d in D] + [f"Proc_{p}" for p in P] + [f"Med_{m}" for m in M]
                visit_data = self.data[visit]
                cdt_discovery = GES()
                #cdt_discovery = GIES()
                causal_graph = cdt_discovery.predict(visit_data)

                new_graph = nx.DiGraph()
                for node in causal_graph.nodes():
                    new_graph.add_node(node)
                for edge in causal_graph.edges():
                    source, target = edge
                    # Reserve disease-medicine, disease-disease, medicine-medicine
                    if source.startswith("Diag") and target.startswith("Diag"):
                        new_graph.add_edge(source, target)
                    elif source.startswith("Diag") and target.startswith("Med"):
                        new_graph.add_edge(source, target)
                    elif source.startswith("Diag") and target.startswith("Proc"):
                        new_graph.add_edge(source, target)
                    elif source.startswith("Proc") and target.startswith("Proc"):
                        new_graph.add_edge(source, target)
                    elif source.startswith("Proc") and target.startswith("Diag"):
                        new_graph.add_edge(source, target)
                    elif source.startswith("Proc") and target.startswith("Med"):
                        new_graph.add_edge(source, target)
                    elif source.startswith("Med") and target.startswith("Med"):
                        new_graph.add_edge(source, target)

                causal_graph = new_graph

                # remove loop
                while not nx.is_directed_acyclic_graph(causal_graph):
                    cycle_nodes = nx.find_cycle(causal_graph, orientation="original")

                    for edge in cycle_nodes:
                        source, target, _ = edge
                        causal_graph.remove_edge(source, target)

                causal_graph = nx.DiGraph(causal_graph)
                causal_graphs.append(causal_graph)

            # d-d p-p m-m
            subgraph_list = []
            for graph in tqdm(causal_graphs):
                graph_type = []

                nodes_to_remove = [node for node in graph.nodes() if "Med" in node or "Proc" in node]
                graph2 = graph.copy()
                graph2.remove_nodes_from(nodes_to_remove)
                graph_type.append(graph2)

                nodes_to_remove = [node for node in graph.nodes() if "Diag" in node or "Med" in node]
                graph2 = graph.copy()
                graph2.remove_nodes_from(nodes_to_remove)
                graph_type.append(graph2)

                nodes_to_remove = [node for node in graph.nodes() if "Diag" in node or "Proc" in node]
                graph2 = graph.copy()
                graph2.remove_nodes_from(nodes_to_remove)
                graph_type.append(graph2)

                subgraph_list.append(graph_type)

            dill.dump(subgraph_list, open(file_path, "wb"))
        return subgraph_list

    def sessions_process(self, raw_data):
        sessions = []
        for patient in raw_data:
            for adm in patient:
                sessions.append(adm)
        return sessions

    def data_process(self, data_train):
        # Get the directory where the current script file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Build the complete file path
        file_path = os.path.join(current_dir, f"../../data/{self.dataset}/graphs/matrix4causalgraph.pkl")
        try:
            with open(file_path, "rb") as f:
                df = dill.load(f)
        except FileNotFoundError:

            print("整理数据集..")
            train_sessions = self.sessions_process(data_train)

            df = pd.DataFrame(0.0, index=range(len(train_sessions)), columns=
            [f'Diag_{i}' for i in range(self.num_d)] +
            [f'Proc_{i}' for i in range(self.num_p)] +
            [f'Med_{i}' for i in range(self.num_m)])

            for i, session in tqdm(enumerate(train_sessions)):
                D, P, M, _ = session
                df.loc[i, [f'Diag_{d}' for d in D]] = 1
                df.loc[i, [f'Proc_{p}' for p in P]] = 1
                df.loc[i, [f'Med_{m}' for m in M]] = 1

            with open(file_path, "wb") as f:
                dill.dump(df, f)
        return df


