import pickle
import os
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader

def load_raw_data(filepath: str):
    """Loads processed subgraph pickle files."""
    paths = {
        'train': os.path.join(filepath, 'train_pg_subgraph.pkl'),
        'val': os.path.join(filepath, 'val_pg_subgraph.pkl'),
        'test': os.path.join(filepath, 'test_pg_subgraph.pkl')
    }
    
    print(f"Loading data from: {filepath}")
    try:
        datasets = {}
        for key, path in paths.items():
            with open(path, 'rb') as f:
                datasets[key] = pickle.load(f)
        return datasets['train'], datasets['val'], datasets['test']
    except FileNotFoundError as e:
        print(f"\n[ERROR] Files not found in {filepath}. Check path configuration.\n")
        raise e

def process_labels(train_pg, val_pg, test_pg):
    """Maps gene IDs to class indices and verifies class distribution."""
    all_true_gene_ids = []
    for dataset in [train_pg, val_pg, test_pg]:
        for p in dataset:
            all_true_gene_ids.extend(p.true_gene_ids)

    unique_true_gene_ids = set(all_true_gene_ids)
    gene_id_mapping = {gene_id: idx for idx, gene_id in enumerate(unique_true_gene_ids)}
    
    # Diagnostic Check
    train_classes = {gene_id_mapping[gid] for p in train_pg for gid in p.true_gene_ids}
    test_classes = {gene_id_mapping[gid] for p in test_pg for gid in p.true_gene_ids}
    overlap = train_classes.intersection(test_classes)
    
    print(f"\n--- DATASET DIAGNOSTICS ---")
    print(f"Unique Classes: {len(unique_true_gene_ids)}")
    print(f"Train/Test Overlap: {len(overlap)} classes")
    if len(overlap) == 0:
        print("[WARNING] Zero-shot setting detected (Disjoint Train/Test classes).")
    print("---------------------------\n")

    for dataset in [train_pg, val_pg, test_pg]:
        for patient in dataset:
            patient.y = torch.tensor([gene_id_mapping[gid] for gid in patient.true_gene_ids], dtype=torch.long)
            
    return train_pg, val_pg, test_pg, len(unique_true_gene_ids)

def preprocess_graph_data(dataset):
    """Standardizes PyG Data objects."""
    processed_graphs = []
    for data in dataset:
        new_data = Data(
            edge_index=data.edge_index, 
            y=data.y, 
            x=data.x,
            original_ids=data.original_ids, 
            edge_attr=data.edge_attr
        )
        processed_graphs.append(new_data)
    return processed_graphs

def optimized_collate_fn(batch):
    """Custom collation for variable-size graphs in a batch."""
    batch_size = len(batch)
    cumsum_nodes = 0
    adjusted_edge_indices = []
    
    for data in batch:
        edge_index = data.edge_index + cumsum_nodes
        adjusted_edge_indices.append(edge_index)
        cumsum_nodes += data.num_nodes

    x = torch.cat([data.x for data in batch], dim=0)
    y = torch.cat([data.y for data in batch], dim=0)
    edge_index = torch.cat(adjusted_edge_indices, dim=1)
    
    edge_attr = None
    if batch[0].edge_attr is not None:
        edge_attr = torch.cat([data.edge_attr for data in batch], dim=0)

    batch_tensor = torch.cat([torch.full((data.num_nodes,), i, dtype=torch.long) for i, data in enumerate(batch)])
    
    original_ids = []
    for data in batch:
        if data.original_ids is not None:
            ids = data.original_ids if not isinstance(data.original_ids, list) else torch.tensor(data.original_ids, dtype=torch.long)
            original_ids.append(ids)
    original_ids = torch.cat(original_ids) if original_ids else None
    
    return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, batch=batch_tensor, original_ids=original_ids, batch_size=batch_size)

def get_dataloaders(data_path: str, batch_size_train: int = 128, batch_size_val: int = 128):
    train_raw, val_raw, test_raw = load_raw_data(data_path)
    train_raw, val_raw, test_raw, num_classes = process_labels(train_raw, val_raw, test_raw)
    
    train_data = preprocess_graph_data(train_raw)
    val_data = preprocess_graph_data(val_raw)
    test_data = preprocess_graph_data(test_raw)
    
    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True, collate_fn=optimized_collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size_val, collate_fn=optimized_collate_fn)
    test_loader = DataLoader(test_data, batch_size=128, collate_fn=optimized_collate_fn)
    
    return train_loader, val_loader, test_loader, num_classes