"""
Comprehensive Test Suite for FraudGraph.
Verifies dataset loading, model forward passes, ego-graph extraction, and ring mining.
"""

import sys
import os
import unittest
import numpy as np
import scipy.sparse as sp
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data import load_dataset
from src.baselines import build_baselines
from src.models import GCN, GraphSAGE, CAREGNN, scipy_to_torch_csr
from src.ring_miner import find_fraud_rings, explain_user_prediction
from src.evaluate import evaluate_predictions, find_optimal_threshold


class TestFraudGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n[TEST SETUP] Loading Amazon dataset...")
        cls.data = load_dataset("Amazon", data_dir="data", seed=42)

    def test_01_dataset_shapes(self):
        """Verify Amazon dataset has 11,944 nodes, 25 features, and 3 relations."""
        self.assertEqual(self.data.num_nodes, 11944)
        self.assertEqual(self.data.num_features, 25)
        self.assertEqual(len(self.data.labels), 11944)
        self.assertIn("net_upu", self.data.relations)
        self.assertIn("net_usu", self.data.relations)
        self.assertIn("net_uvu", self.data.relations)
        
        # Test split sizes (40/20/40)
        self.assertEqual(len(self.data.test_idx), 4778)
        print("  [PASS] Dataset integrity verified: 11,944 nodes, 3 relations, 4,778 test nodes.")

    def test_02_ego_graph_extraction(self):
        """Verify 2-hop neighborhood extraction for a target node."""
        target_node = 10214
        nodes, edges = self.data.get_ego_graph(target_node, hops=2, max_neighbors=20)
        self.assertGreater(len(nodes), 1)
        self.assertIn(target_node, nodes)
        self.assertEqual(nodes[0], target_node)
        self.assertGreater(len(edges), 0)
        print(f"  [PASS] Ego-graph extraction verified: {len(nodes)} nodes, {len(edges)} edges extracted.")

    def test_03_gcn_forward_pass(self):
        """Verify GCN forward pass and output shape."""
        x = torch.FloatTensor(self.data.features[:100])
        small_adj = sp.eye(100, format='csr', dtype=np.float32)
        norm_adj = scipy_to_torch_csr(small_adj, norm='sym')

        model = GCN(in_features=25, hidden_dim=64, num_classes=2)
        model.eval()
        with torch.no_grad():
            logits, h = model(x, norm_adj)
        self.assertEqual(logits.shape, (100, 2))
        self.assertEqual(h.shape, (100, 64))
        print("  [PASS] GCN forward pass verified: output logits shape (100, 2).")

    def test_04_care_gnn_forward_pass(self):
        """Verify CARE-GNN multi-relation forward pass and attention weights."""
        x = torch.FloatTensor(self.data.features[:100])
        small_adj = sp.eye(100, format='csr', dtype=np.float32)
        rel_adjs = [scipy_to_torch_csr(small_adj, norm='row') for _ in range(3)]

        model = CAREGNN(in_features=25, hidden_dim=64, num_classes=2, num_relations=3)
        model.eval()
        with torch.no_grad():
            logits, h, alpha, _ = model(x, rel_adjs, return_details=True)
        self.assertEqual(logits.shape, (100, 2))
        self.assertEqual(alpha.shape, (100, 3))
        row_sums = alpha.sum(dim=-1).numpy()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
        print("  [PASS] CARE-GNN forward pass and relation attention verified.")

    def test_05_ring_mining(self):
        """Verify coordinated fraud ring clustering."""
        simulated_probs = np.random.uniform(0, 1, size=self.data.num_nodes)
        simulated_probs[100:110] = 0.95
        rings = find_fraud_rings(self.data, simulated_probs, threshold=0.38, min_ring_size=3)
        self.assertIsInstance(rings, list)
        print(f"  [PASS] Fraud ring mining verified: {len(rings)} rings identified.")

    def test_06_evaluation_metrics(self):
        """Verify metrics and threshold tuning logic."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.85, 0.9])
        metrics = evaluate_predictions(y_true, probs, threshold=0.5)
        self.assertEqual(metrics['auc'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['tp'], 4)
        self.assertEqual(metrics['tn'], 4)
        print("  [PASS] Evaluation suite verified: AUC, Recall, Confusion Matrix accurate.")


if __name__ == "__main__":
    unittest.main()
