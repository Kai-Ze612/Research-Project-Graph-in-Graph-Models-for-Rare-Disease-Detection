import pickle
import os
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader

def load_raw_data(filepath):
    """Loads the pickle files from disk."""
    train_path = os.path.join(filepath, 'train_pg_subgraph.pkl')
    val_path = os.path.join(filepath, 'val_pg_subgraph.pkl')
    test_path = os.path.join(filepath, 'test_pg_subgraph.pkl')
    
    with open(train_path, 'rb') as file:   
        train_pg = pickle.load(file)
    with open(val_path, 'rb') as file:
        val_pg = pickle.load(file)
    with open(test_path, 'rb') as file:
        test_pg = pickle.load(file)

    return train_pg, val_pg, test_pg

def process_labels(train_pg, val_pg, test_pg):
    """Maps unique gene IDs to class indices 0-N and checks for data leakage/overlap."""
    all_true_gene_ids = []
    for patient in train_pg: all_true_gene_ids.extend(patient.true_gene_ids)
    for patient in val_pg: all_true_gene_ids.extend(patient.true_gene_ids)
    for patient in test_pg: all_true_gene_ids.extend(patient.true_gene_ids)

    # create mapping
    unique_true_gene_ids = set(all_true_gene_ids)
    gene_id_mapping = {gene_id: idx for idx, gene_id in enumerate(unique_true_gene_ids)}
    
    # --- DEBUG: CHECK FOR OVERLAP ---
    # We need to know if the classes in Test set actually exist in Train set
    train_classes = set()
    for p in train_pg:
        for gid in p.true_gene_ids:
            train_classes.add(gene_id_mapping[gid])
            
    test_classes = set()
    for p in test_pg:
        for gid in p.true_gene_ids:
            test_classes.add(gene_id_mapping[gid])
    
    overlap = train_classes.intersection(test_classes)
    
    print(f"\n--- DATASET DIAGNOSTICS ---")
    print(f"Total Unique Classes (Diseases): {len(unique_true_gene_ids)}")
    print(f"Classes in Train Set: {len(train_classes)}")
    print(f"Classes in Test Set: {len(test_classes)}")
    print(f"Overlapping Classes: {len(overlap)}")
    
    if len(overlap) == 0:
        print("\n[CRITICAL WARNING] Test set has completely DIFFERENT classes than Train set.")
        print("The model will have 0.0% accuracy because it is predicting unseen labels.")
        print("Standard classification cannot handle zero-shot transfer without specific design.")
    elif len(overlap) < len(test_classes):
        print(f"\n[WARNING] {len(test_classes) - len(overlap)} classes in Test set are NOT in Train set.")
        print("Accuracy will be capped because these classes are impossible to predict.")
    else:
        print("OK: All Test classes are present in Training data.")
    print("---------------------------\n")
    # --------------------------------

    # Assign new y labels
    for dataset in [train_pg, val_pg, test_pg]:
        for patient in dataset:
            patient.y = torch.tensor([gene_id_mapping[gid] for gid in patient.true_gene_ids], dtype=torch.long)
            
    return train_pg, val_pg, test_pg, len(unique_true_gene_ids)

def preprocess_graph_data(dataset):
    """Extracts only relevant fields into new Data objects."""
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
    """Custom collate function for handling multiple graphs in a batch."""
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
            ids = data.original_ids
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long)
            original_ids.append(ids)
    original_ids = torch.cat(original_ids) if original_ids else None
    
    return Data(
        x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
        batch=batch_tensor, original_ids=original_ids, batch_size=batch_size
    )

def get_dataloaders(data_path, batch_size_train=1024, batch_size_val=512):
    """Main function to get loaders and num_classes."""
    train_raw, val_raw, test_raw = load_raw_data(data_path)
    train_raw, val_raw, test_raw, num_classes = process_labels(train_raw, val_raw, test_raw)
    
    train_data = preprocess_graph_data(train_raw)
    val_data = preprocess_graph_data(val_raw)
    test_data = preprocess_graph_data(test_raw)
    
    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True, collate_fn=optimized_collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size_val, collate_fn=optimized_collate_fn)
    test_loader = DataLoader(test_data, batch_size=128, collate_fn=optimized_collate_fn)
    
    return train_loader, val_loader, test_loader, num_classes