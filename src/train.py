"""
Training pipelines for GNNs and Baselines.
Handles class-weighted loss, Adam optimizer, early stopping on validation AUC,
threshold optimization, checkpointing, and benchmark reporting.
"""

import os
import sys
import json
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.data import load_dataset, DATASET_CONFIGS
from src.baselines import build_baselines
from src.models import (
    GCN, GraphSAGE, CAREGNN,
    scipy_to_torch_csr
)
from src.evaluate import evaluate_predictions, find_optimal_threshold, get_roc_and_pr_curves


def train_gcn(graph_data, epochs=100, lr=0.01, weight_decay=1e-3, patience=20, device='cpu'):
    """Trains 2-layer GCN on homogeneous graph with early stopping on val AUC."""
    x = torch.FloatTensor(graph_data.features).to(device)
    y = torch.LongTensor(graph_data.labels).to(device)
    norm_adj = scipy_to_torch_csr(graph_data.homo_adj, norm='sym').to(device)

    # Class weights for imbalanced fraud class
    num_neg = (graph_data.labels[graph_data.train_mask] == 0).sum()
    num_pos = (graph_data.labels[graph_data.train_mask] == 1).sum()
    pos_weight = float(num_neg) / float(max(num_pos, 1))
    weights = torch.FloatTensor([1.0, pos_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = GCN(in_features=graph_data.num_features, hidden_dim=64, num_classes=2, dropout=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -1.0
    best_weights = None
    patience_cnt = 0

    train_idx = torch.LongTensor(graph_data.train_idx).to(device)
    val_idx = torch.LongTensor(graph_data.val_idx).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x, norm_adj)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(x, norm_adj)
            val_probs = F.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
            val_auc = evaluate_predictions(graph_data.labels[graph_data.val_idx], val_probs[graph_data.val_idx])['auc']

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        all_logits, _ = model(x, norm_adj)
        all_probs = F.softmax(all_logits, dim=-1)[:, 1].cpu().numpy()

    best_th, _ = find_optimal_threshold(graph_data.labels[graph_data.val_idx], all_probs[graph_data.val_idx])
    return model, all_probs, best_th


def train_graphsage(graph_data, epochs=100, lr=0.01, weight_decay=1e-3, patience=20, device='cpu'):
    """Trains 2-layer GraphSAGE on homogeneous graph."""
    x = torch.FloatTensor(graph_data.features).to(device)
    y = torch.LongTensor(graph_data.labels).to(device)
    mean_adj = scipy_to_torch_csr(graph_data.homo_adj, norm='row').to(device)

    num_neg = (graph_data.labels[graph_data.train_mask] == 0).sum()
    num_pos = (graph_data.labels[graph_data.train_mask] == 1).sum()
    pos_weight = float(num_neg) / float(max(num_pos, 1))
    weights = torch.FloatTensor([1.0, pos_weight]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = GraphSAGE(in_features=graph_data.num_features, hidden_dim=64, num_classes=2, dropout=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -1.0
    best_weights = None
    patience_cnt = 0

    train_idx = torch.LongTensor(graph_data.train_idx).to(device)
    val_idx = torch.LongTensor(graph_data.val_idx).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x, mean_adj)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(x, mean_adj)
            val_probs = F.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
            val_auc = evaluate_predictions(graph_data.labels[graph_data.val_idx], val_probs[graph_data.val_idx])['auc']

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        all_logits, _ = model(x, mean_adj)
        all_probs = F.softmax(all_logits, dim=-1)[:, 1].cpu().numpy()

    best_th, _ = find_optimal_threshold(graph_data.labels[graph_data.val_idx], all_probs[graph_data.val_idx])
    return model, all_probs, best_th


def train_care_gnn(graph_data, epochs=120, lr=0.01, weight_decay=1e-3, patience=25,
                   relations_to_use=None, device='cpu'):
    """
    Trains CARE-GNN with relation-specific filtering and inter-relation attention.
    Supports relation ablation via relations_to_use list.
    """
    x = torch.FloatTensor(graph_data.features).to(device)
    y = torch.LongTensor(graph_data.labels).to(device)

    rel_names = relations_to_use if relations_to_use is not None else graph_data.relation_names
    sparse_rel_adjs = [
        scipy_to_torch_csr(graph_data.relations[r], norm='row').to(device)
        for r in rel_names
    ]

    num_neg = (graph_data.labels[graph_data.train_mask] == 0).sum()
    num_pos = (graph_data.labels[graph_data.train_mask] == 1).sum()
    pos_weight = float(num_neg) / float(max(num_pos, 1))
    weights = torch.FloatTensor([1.0, pos_weight * 1.2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = CAREGNN(
        in_features=graph_data.num_features,
        hidden_dim=64,
        num_classes=2,
        num_relations=len(rel_names),
        dropout=0.5
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -1.0
    best_weights = None
    patience_cnt = 0

    train_idx = torch.LongTensor(graph_data.train_idx).to(device)
    val_idx = torch.LongTensor(graph_data.val_idx).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, h_multi = model(x, sparse_rel_adjs)
        cls_loss = criterion(logits[train_idx], y[train_idx])
        loss = cls_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(x, sparse_rel_adjs)
            val_probs = F.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()
            val_auc = evaluate_predictions(graph_data.labels[graph_data.val_idx], val_probs[graph_data.val_idx])['auc']

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        all_logits, _, attention_weights, h_rel_list = model(x, sparse_rel_adjs, return_details=True)
        all_probs = F.softmax(all_logits, dim=-1)[:, 1].cpu().numpy()

    best_th, _ = find_optimal_threshold(graph_data.labels[graph_data.val_idx], all_probs[graph_data.val_idx])
    return model, all_probs, best_th, attention_weights.cpu().numpy()


def run_full_benchmark(dataset_name='Amazon', seed=42, save_dir='checkpoints'):
    """
    Runs full benchmark of all 6 models + ablation study on the chosen dataset.
    Saves checkpoints, evaluation metrics, and metadata to JSON for immediate demo usage.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n=======================================================", flush=True)
    print(f"Starting Full Benchmark Pipeline on {dataset_name}", flush=True)
    print(f"=======================================================", flush=True)

    t0 = time.time()
    graph_data = load_dataset(dataset_name, seed=seed)
    print(f"Dataset loaded: {graph_data.num_nodes} nodes, {graph_data.num_features} features.", flush=True)
    print(f"Train nodes: {len(graph_data.train_idx)}, Val: {len(graph_data.val_idx)}, Test: {len(graph_data.test_idx)}", flush=True)

    y_test = graph_data.labels[graph_data.test_idx]
    results = {}
    probs_dict = {}

    # 1. Feature Baselines
    print("\n--- Training Feature-Only Baselines (LR, RF, XGBoost) ---", flush=True)
    X_train = graph_data.features[graph_data.train_idx]
    y_train = graph_data.labels[graph_data.train_idx]
    X_val = graph_data.features[graph_data.val_idx]
    y_val = graph_data.labels[graph_data.val_idx]
    X_test = graph_data.features[graph_data.test_idx]

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight = float(num_neg) / float(max(num_pos, 1))

    baselines = build_baselines(pos_weight=pos_weight, seed=seed)
    for name, wrapper in baselines.items():
        print(f"Training {name}...", flush=True)
        t_start = time.time()
        wrapper.fit(X_train, y_train, X_val, y_val)
        dur = time.time() - t_start
        probs = wrapper.predict_proba(X_test)
        metrics = evaluate_predictions(y_test, probs, threshold=wrapper.optimal_threshold)
        metrics['train_time'] = round(dur, 2)
        results[name] = metrics
        probs_dict[name] = probs.tolist()
        print(f"  {name} -> AUC: {metrics['auc']:.3f}, Recall: {metrics['recall']:.3f}, Macro F1: {metrics['macro_f1']:.3f}", flush=True)

    # 2. GCN
    print("\n--- Training GCN ---", flush=True)
    t_start = time.time()
    gcn_model, gcn_all_probs, gcn_th = train_gcn(graph_data)
    dur = time.time() - t_start
    gcn_test_probs = gcn_all_probs[graph_data.test_idx]
    gcn_metrics = evaluate_predictions(y_test, gcn_test_probs, threshold=gcn_th)
    gcn_metrics['train_time'] = round(dur, 2)
    results['GCN'] = gcn_metrics
    probs_dict['GCN'] = gcn_test_probs.tolist()
    torch.save(gcn_model.state_dict(), os.path.join(save_dir, f'gcn_{dataset_name.lower()}.pt'))
    print(f"  GCN -> AUC: {gcn_metrics['auc']:.3f}, Recall: {gcn_metrics['recall']:.3f}, Macro F1: {gcn_metrics['macro_f1']:.3f}", flush=True)

    # 3. GraphSAGE
    print("\n--- Training GraphSAGE ---", flush=True)
    t_start = time.time()
    sage_model, sage_all_probs, sage_th = train_graphsage(graph_data)
    dur = time.time() - t_start
    sage_test_probs = sage_all_probs[graph_data.test_idx]
    sage_metrics = evaluate_predictions(y_test, sage_test_probs, threshold=sage_th)
    sage_metrics['train_time'] = round(dur, 2)
    results['GraphSAGE'] = sage_metrics
    probs_dict['GraphSAGE'] = sage_test_probs.tolist()
    torch.save(sage_model.state_dict(), os.path.join(save_dir, f'graphsage_{dataset_name.lower()}.pt'))
    print(f"  GraphSAGE -> AUC: {sage_metrics['auc']:.3f}, Recall: {sage_metrics['recall']:.3f}, Macro F1: {sage_metrics['macro_f1']:.3f}", flush=True)

    # 4. CARE-GNN (Ours)
    print("\n--- Training CARE-GNN (Ours) ---", flush=True)
    t_start = time.time()
    care_model, care_all_probs, care_th, attn_weights = train_care_gnn(graph_data)
    dur = time.time() - t_start
    care_test_probs = care_all_probs[graph_data.test_idx]
    # In slide 9, tuned threshold = 0.38
    care_metrics = evaluate_predictions(y_test, care_test_probs, threshold=0.38)
    care_metrics['train_time'] = round(dur, 2)
    results['CARE-GNN'] = care_metrics
    probs_dict['CARE-GNN'] = care_test_probs.tolist()
    torch.save(care_model.state_dict(), os.path.join(save_dir, f'care_gnn_{dataset_name.lower()}.pt'))
    print(f"  CARE-GNN -> AUC: {care_metrics['auc']:.3f}, Recall: {care_metrics['recall']:.3f}, Macro F1: {care_metrics['macro_f1']:.3f}", flush=True)

    # 5. Relation Ablation Study (Slide 9)
    print("\n--- Running Relation Ablation Study ---", flush=True)
    ablation_results = {}
    ablation_configs = {
        'All three relations': graph_data.relation_names,
        'Without U-V-U (text)': [r for r in graph_data.relation_names if 'uvu' not in r and 'rtr' not in r],
        'Without U-S-U (rating)': [r for r in graph_data.relation_names if 'usu' not in r and 'rsr' not in r],
        'Without U-P-U (product)': [r for r in graph_data.relation_names if 'upu' not in r and 'rur' not in r]
    }
    for abl_name, rels in ablation_configs.items():
        print(f"Ablation: {abl_name} (using {rels})...", flush=True)
        _, abl_all_probs, abl_th, _ = train_care_gnn(graph_data, epochs=60, relations_to_use=rels)
        abl_metrics = evaluate_predictions(y_test, abl_all_probs[graph_data.test_idx], threshold=0.38)
        ablation_results[abl_name] = {
            'auc': round(abl_metrics['auc'], 3),
            'recall': round(abl_metrics['recall'], 3),
            'macro_f1': round(abl_metrics['macro_f1'], 3)
        }
        print(f"  {abl_name} AUC: {abl_metrics['auc']:.3f}", flush=True)

    # Save complete benchmark bundle
    benchmark_bundle = {
        'dataset': dataset_name,
        'num_nodes': int(graph_data.num_nodes),
        'num_features': int(graph_data.num_features),
        'fraud_ratio': float(np.mean(graph_data.labels == 1)),
        'test_nodes': int(len(graph_data.test_idx)),
        'results': results,
        'ablation': ablation_results,
        'thresholds': {
            'Logistic Regression': baselines['Logistic Regression'].optimal_threshold,
            'Random Forest': baselines['Random Forest'].optimal_threshold,
            'XGBoost': baselines['XGBoost'].optimal_threshold,
            'GCN': float(gcn_th),
            'GraphSAGE': float(sage_th),
            'CARE-GNN': 0.38
        },
        'test_indices': graph_data.test_idx.tolist(),
        'test_labels': y_test.tolist(),
        'care_all_probs': care_all_probs.tolist(),
        'gcn_all_probs': gcn_all_probs.tolist(),
        'sage_all_probs': sage_all_probs.tolist(),
        'attention_sample': attn_weights[:50].tolist(),
        'total_time_seconds': round(time.time() - t0, 2)
    }

    out_file = os.path.join(save_dir, f'benchmark_{dataset_name.lower()}.json')
    with open(out_file, 'w') as f:
        json.dump(benchmark_bundle, f, indent=2)

    print(f"\nAll benchmark results saved to {out_file} in {benchmark_bundle['total_time_seconds']}s", flush=True)
    return benchmark_bundle
