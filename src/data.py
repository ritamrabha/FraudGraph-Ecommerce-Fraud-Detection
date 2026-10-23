"""
Data loader and preprocessing for Amazon and YelpChi fraud datasets.
Handles multi-relational graphs, feature standardization, and 40/20/40 stratified splits.
"""

import os
import urllib.request
import zipfile
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


DATASET_URLS = {
    'Amazon': 'https://data.dgl.ai/dataset/FraudAmazon.zip',
    'YelpChi': 'https://data.dgl.ai/dataset/FraudYelp.zip'
}

DATASET_CONFIGS = {
    'Amazon': {
        'filename': 'Amazon.mat',
        'relations': ['net_upu', 'net_usu', 'net_uvu'],
        'relation_labels': {
            'net_upu': 'U-P-U (User-Product-User)',
            'net_usu': 'U-S-U (User-Star-User: Same Rating/Week)',
            'net_uvu': 'U-V-U (User-Text-User: Review Similarity)'
        },
        'node_type': 'User',
        'feature_names': [
            'Rev_Count', 'Avg_Rating', 'Rating_Entropy', 'Verified_Ratio',
            'Avg_Helpful_Votes', 'Helpful_Ratio', 'Day_Span', 'Rev_Rate',
            'Text_Len_Avg', 'Text_Entropy', 'Caps_Ratio', 'Punct_Ratio',
            'Sentiment_Score', 'Extreme_Rating_Ratio', 'Same_Day_Rev_Ratio',
            'Burstiness_Score', 'Prod_Category_Entropy', 'Price_Variance',
            'Review_Gap_Std', 'First_Reviewer_Ratio', 'Max_Daily_Rev',
            'Avg_Summary_Len', 'Rating_Deviation_Mean', 'Early_Reviewer_Flag', 'Camouflage_Risk'
        ]
    },
    'YelpChi': {
        'filename': 'YelpChi.mat',
        'relations': ['net_rur', 'net_rsr', 'net_rtr'],
        'relation_labels': {
            'net_rur': 'R-U-R (Review-User-Review)',
            'net_rsr': 'R-S-R (Review-Star-Review)',
            'net_rtr': 'R-T-R (Review-Time-Review)'
        },
        'node_type': 'Review',
        'feature_names': [f'Feature_{i+1}' for i in range(32)]
    }
}


class GraphData:
    """Container for multi-relational graph fraud dataset."""
    def __init__(self, name, features, labels, relations, homo_adj,
                 train_mask, val_mask, test_mask, feature_names=None):
        self.name = name
        self.features = features  # (N, D) float32 numpy array
        self.labels = labels      # (N,) int64 numpy array
        self.relations = relations  # dict of relation_name -> scipy.sparse.csr_matrix
        self.homo_adj = homo_adj    # scipy.sparse.csr_matrix combined graph
        self.train_mask = train_mask  # boolean array (N,)
        self.val_mask = val_mask      # boolean array (N,)
        self.test_mask = test_mask    # boolean array (N,)
        self.feature_names = feature_names or [f'feat_{i}' for i in range(features.shape[1])]
        
        self.num_nodes = features.shape[0]
        self.num_features = features.shape[1]
        self.num_classes = 2
        self.relation_names = list(relations.keys())

    @property
    def train_idx(self):
        return np.where(self.train_mask)[0]

    @property
    def val_idx(self):
        return np.where(self.val_mask)[0]

    @property
    def test_idx(self):
        return np.where(self.test_mask)[0]

    def get_ego_graph(self, node_id, hops=2, max_neighbors=30):
        """
        Extract the 1-hop or 2-hop neighborhood of a given node for visualization and inspection.
        Returns:
            sub_nodes: list of unique node ids (target node is first)
            sub_edges: list of dicts {source, target, relation, weight}
        """
        current_layer = {node_id}
        visited = {node_id}
        edges_collected = []

        for h in range(hops):
            next_layer = set()
            for u in current_layer:
                # Iterate across each relation
                for rel_name, adj in self.relations.items():
                    # get row u neighbors
                    row = adj.getrow(u)
                    neighbors = row.indices
                    if len(neighbors) == 0:
                        continue
                    # sample if too many neighbors to prevent browser DOM overload
                    if len(neighbors) > max_neighbors:
                        chosen = np.random.choice(neighbors, max_neighbors, replace=False)
                    else:
                        chosen = neighbors
                    
                    for v in chosen:
                        if v != u:
                            edges_collected.append({
                                'source': int(u),
                                'target': int(v),
                                'relation': rel_name
                            })
                            if v not in visited:
                                visited.add(v)
                                next_layer.add(v)
            current_layer = next_layer
            if not current_layer:
                break

        sub_nodes = list(visited)
        if node_id in sub_nodes:
            sub_nodes.remove(node_id)
            sub_nodes.insert(0, node_id)

        # Deduplicate edges (undirected)
        unique_edges = []
        edge_keys = set()
        for e in edges_collected:
            src, tgt, rel = e['source'], e['target'], e['relation']
            canon_key = (min(src, tgt), max(src, tgt), rel)
            if canon_key not in edge_keys:
                edge_keys.add(canon_key)
                unique_edges.append(e)

        return sub_nodes, unique_edges


def download_and_extract_if_needed(name='Amazon', data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    cfg = DATASET_CONFIGS[name]
    mat_path = os.path.join(data_dir, cfg['filename'])

    if not os.path.exists(mat_path):
        url = DATASET_URLS[name]
        zip_path = os.path.join(data_dir, f"{name}.zip")
        print(f"Downloading {name} dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"Dataset ready at {mat_path}")
    return mat_path


def load_dataset(name='Amazon', data_dir='data', train_ratio=0.4, val_ratio=0.2, test_ratio=0.4, seed=42):
    """
    Load dataset from .mat file and perform 40/20/40 stratified split.
    """
    mat_path = download_and_extract_if_needed(name, data_dir)
    data = sio.loadmat(mat_path)
    cfg = DATASET_CONFIGS[name]

    # Extract features and convert to dense float32
    raw_feats = data['features']
    if sp.issparse(raw_feats):
        raw_feats = raw_feats.toarray()
    raw_feats = raw_feats.astype(np.float32)

    # Standardize features (mean=0, std=1)
    scaler = StandardScaler()
    norm_feats = scaler.fit_transform(raw_feats).astype(np.float32)

    # Extract labels
    labels = data['label'].flatten().astype(np.int64)

    # Extract relations
    relations = {}
    for rel in cfg['relations']:
        mat = data[rel]
        if not sp.issparse(mat):
            mat = sp.csr_matrix(mat)
        else:
            mat = mat.tocsr()
        # Symmetrize and remove self-loops
        mat = mat + mat.T
        mat = (mat > 0).astype(np.float32)
        mat.setdiag(0)
        mat.eliminate_zeros()
        relations[rel] = mat

    # Build or extract homogeneous graph
    if 'homo' in data:
        homo_adj = data['homo']
        if not sp.issparse(homo_adj):
            homo_adj = sp.csr_matrix(homo_adj)
        else:
            homo_adj = homo_adj.tocsr()
    else:
        # Combine all relation graphs
        combined = sp.csr_matrix(relations[cfg['relations'][0]].shape, dtype=np.float32)
        for rel in cfg['relations']:
            combined = combined + relations[rel]
        homo_adj = (combined > 0).astype(np.float32)

    homo_adj = homo_adj + homo_adj.T
    homo_adj = (homo_adj > 0).astype(np.float32)
    homo_adj.setdiag(0)
    homo_adj.eliminate_zeros()

    # Stratified 40 / 20 / 40 split (Train: 40%, Val: 20%, Test: 40%)
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels
    )
    val_rel_ratio = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_rel_ratio,
        random_state=seed,
        stratify=labels[temp_idx]
    )

    num_nodes = len(labels)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    feature_names = cfg['feature_names']
    # Ensure length matches feature dimension
    if len(feature_names) != norm_feats.shape[1]:
        feature_names = [f'Feat_{i+1}' for i in range(norm_feats.shape[1])]

    graph_data = GraphData(
        name=name,
        features=norm_feats,
        labels=labels,
        relations=relations,
        homo_adj=homo_adj,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        feature_names=feature_names
    )
    return graph_data
