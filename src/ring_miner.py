"""
Fraud Ring Mining and Explainability Utilities.
Detects coordinated sybil attacks, clusters of colluding reviewers,
and generates local graph explanations.
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp


def find_fraud_rings(graph_data, fraud_probs, threshold=0.38, min_ring_size=3, max_rings=25):
    """
    Finds coordinated fraud rings (connected subgraphs of high-risk nodes).
    Uses the multi-relational graph (especially rating collision U-S-U / R-S-R).
    """
    # Identify high risk nodes
    high_risk_mask = fraud_probs >= threshold
    high_risk_nodes = set(np.where(high_risk_mask)[0])

    if len(high_risk_nodes) == 0:
        return []

    # Build NetworkX graph from U-S-U (strongest fraud signal) and U-V-U
    G = nx.Graph()
    for u in high_risk_nodes:
        G.add_node(u)

    # Add edges between high risk nodes
    primary_rel = 'net_usu' if 'net_usu' in graph_data.relations else graph_data.relation_names[0]
    adj = graph_data.relations[primary_rel]

    for u in high_risk_nodes:
        row = adj.getrow(u)
        for v in row.indices:
            if v in high_risk_nodes and v > u:
                G.add_edge(u, v, relation=primary_rel)

    # Also add review text similarity edges if present
    text_rel = 'net_uvu' if 'net_uvu' in graph_data.relations else None
    if text_rel and text_rel in graph_data.relations:
        text_adj = graph_data.relations[text_rel]
        for u in high_risk_nodes:
            row = text_adj.getrow(u)
            for v in row.indices:
                if v in high_risk_nodes and v > u:
                    G.add_edge(u, v, relation=text_rel)

    # Extract connected components
    components = list(nx.connected_components(G))
    rings = []

    for comp_idx, comp in enumerate(components):
        if len(comp) >= min_ring_size:
            comp_list = list(comp)
            subg = G.subgraph(comp)
            density = nx.density(subg) if len(comp) > 1 else 1.0
            avg_prob = float(np.mean([fraud_probs[n] for n in comp_list]))
            max_prob = float(np.max([fraud_probs[n] for n in comp_list]))

            # Find representative/hub node
            degrees = dict(subg.degree())
            hub_node = max(degrees, key=degrees.get) if degrees else comp_list[0]

            rings.append({
                'ring_id': f"RING-{comp_idx + 1:03d}",
                'size': len(comp_list),
                'density': round(density, 3),
                'avg_fraud_prob': round(avg_prob, 3),
                'max_fraud_prob': round(max_prob, 3),
                'hub_node': int(hub_node),
                'nodes': [int(n) for n in comp_list],
                'edge_count': subg.number_of_edges(),
                'syndicate_type': 'Coordinated Rating Ring' if density > 0.4 else 'Sparse Review Syndicate'
            })

    # Sort by risk score (size * avg_prob)
    rings.sort(key=lambda r: (r['size'] * r['avg_fraud_prob']), reverse=True)
    return rings[:max_rings]


def explain_user_prediction(graph_data, node_id, fraud_prob, feature_names=None):
    """
    Generates feature and neighborhood explanations for an individual node prediction.
    """
    feats = graph_data.features[node_id]
    labels = graph_data.labels
    
    # 1. Feature deviations from normal
    # Compare with dataset mean
    mean_feats = np.mean(graph_data.features, axis=0)
    std_feats = np.std(graph_data.features, axis=0) + 1e-6
    z_scores = (feats - mean_feats) / std_feats

    names = feature_names or graph_data.feature_names
    top_pos_idx = np.argsort(z_scores)[-4:][::-1]
    top_anomalies = [
        {'feature': names[i], 'value': float(feats[i]), 'z_score': float(z_scores[i])}
        for i in top_pos_idx if z_scores[i] > 0.5
    ]

    # 2. Neighborhood relation breakdown
    rel_counts = {}
    total_neighs = set()
    camouflaged_neighbors = 0

    for rel_name, adj in graph_data.relations.items():
        row = adj.getrow(node_id)
        neighs = row.indices
        rel_counts[rel_name] = len(neighs)
        total_neighs.update(neighs)

    # Check camouflage: genuine neighbors connected to this node
    for n in total_neighs:
        if labels[n] == 0:
            camouflaged_neighbors += 1

    camouflage_ratio = camouflaged_neighbors / max(len(total_neighs), 1)

    return {
        'node_id': int(node_id),
        'fraud_prob': float(fraud_prob),
        'total_neighbors': len(total_neighs),
        'relation_breakdown': rel_counts,
        'camouflaged_genuine_neighbors': int(camouflaged_neighbors),
        'camouflage_ratio': round(camouflage_ratio, 3),
        'top_anomalous_features': top_anomalies,
        'actual_label': int(labels[node_id]),
        'is_test_set': bool(graph_data.test_mask[node_id])
    }
