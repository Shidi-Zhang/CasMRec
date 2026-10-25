"""Small real-model checks; no clinical records or cached causal graphs required."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'src'))
import networkx as nx
import numpy as np
import torch
from modules.HemoMole import HemoMoleModel
from modules.gnn import graph_batch_from_smile
from util import ddi_rate_score
from training import eval_one_epoch


class CausalFixture:
    def get_threshold_effect(self, *args):
        return 0.5

    def get_effect(self, *args):
        return 0.1

    def get_graph(self, index, kind):
        ids = [1, 2] if kind == 'Med' and index == 1 else [0, 1]
        return nx.DiGraph([(f'{kind}_{ids[0]}', f'{kind}_{ids[1]}')])


class ModelChecks(unittest.TestCase):
    def test_ddi_uses_supplied_matrix(self):
        matrix = np.zeros((3, 3))
        matrix[1, 0] = 1
        self.assertAlmostEqual(ddi_rate_score([[[0, 1, 2]]], ddi_adj=matrix), 1 / 3)
        self.assertEqual(ddi_rate_score([[[0]]], ddi_adj=matrix), 0)

    def test_history_branch_receives_gradients_and_evaluates(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        params = dict(num_layer=2, emb_dim=8, graph_pooling='mean',
                      drop_ratio=0.0, gnn_type='gin', virtual_node=False)
        model = HemoMoleModel(params, None, CausalFixture(), 8, [3, 3, 3, 3],
                              4, 8, 8,
                              [np.ones((3, 3)), np.ones((3, 3)), torch.eye(3)],
                              dropout=0, history_query=True)
        drugs = dict(substruct_data=None,
                     mol_data={'batched_data': graph_batch_from_smile(['CC', 'CO', 'CN'])},
                     ddi_mask_H=torch.ones(3, 4), tensor_ddi_adj=torch.zeros(3, 3),
                     average_projection=torch.eye(3))
        patient = [[[0, 1], [0, 1], [0, 1], 0],
                   [[0, 1], [0, 1], [1, 2], 1]]
        score, penalty = model(patient_data=patient, **drugs)
        self.assertTrue(torch.isfinite(score).all())
        (score.square().mean() + penalty).backward()
        for weight in (model.query1[1].weight, model.seq_encoders[2].weight_ih_l0, model.patient_head.weight):
            self.assertIsNotNone(weight.grad)
            self.assertGreater(weight.grad.abs().sum().item(), 0)
        self.assertFalse(torch.equal(model.substruct_emb[0], model.substruct_emb[1]))
        metrics = eval_one_epoch(model, [patient], [3, 3, 3, 3], drugs)
        self.assertEqual(len(metrics), 7)
        self.assertTrue(np.isfinite(metrics).all())
        # Current target medications must not enter prediction inputs.
        import copy
        altered = copy.deepcopy(patient)
        altered[-1][2] = [0]
        with torch.no_grad():
            original_score, _ = model(patient_data=patient, **drugs)
            altered_score, _ = model(patient_data=altered, **drugs)
        self.assertTrue(torch.allclose(original_score, altered_score))
        # Strict checkpoint round trip preserves predictions.
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        with torch.no_grad():
            restored_score, _ = model(patient_data=patient, **drugs)
        self.assertTrue(torch.allclose(original_score, restored_score))



class GraphChecks(unittest.TestCase):
    def test_node_order_does_not_change_weight_assignment(self):
        from modules.Causal_inference import homo_relation_graph
        layer = homo_relation_graph(2, torch.device('cpu'))
        with torch.no_grad():
            layer.causal_weight.list_weights[0].mask_weights.fill_(2)
            layer.causal_weight.list_weights[3].mask_weights.fill_(3)
        graph = nx.DiGraph()
        graph.add_nodes_from(['Med_1', 'Med_0'])
        graph.add_edge('Med_0', 'Med_1')
        result = layer(graph, torch.ones(1, 2, 2), ['Med_0', 'Med_1'])
        self.assertTrue(torch.equal(result, torch.tensor([[[2., 2.], [3., 3.]]])))

    def test_weak_effects_and_empty_graph(self):
        from modules.Causal_inference import hetero_effect_graph
        layer = hetero_effect_graph(2, 2, torch.device('cpu'))
        entity, molecule = torch.randn(1, 2, 2), torch.randn(1, 3, 2)
        weights = np.zeros((2, 3))
        weights[1, 2] = .01
        graph = layer.create_hetero_graph(entity, molecule, weights)
        self.assertEqual(graph['Mole', 'connected__1', 'Entity'].edge_index.tolist(), [[2], [1]])
        output = layer(entity, molecule, np.zeros((2, 3)))
        self.assertEqual(tuple(output.shape), (1, 2, 2))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
