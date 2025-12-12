import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATv2Conv, global_add_pool
from torch_geometric.utils import degree

class F1NodeLevelModule(nn.Module):
    """
    Node-level feature extraction module (F1).
    Processes individual patient graphs to learn node and graph-level representations.
    Incorporates hybrid features (biological + structural) to ensure patient distinguishability.
    """
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, 
                 conv_type: str = 'GCN', dropout: float = 0.3, num_layers: int = 2, heads: int = 4):
        super().__init__()
        self.dropout = dropout
        self.conv_type = conv_type
        
        # Augment input dimension for structural feature (node degree)
        self.feature_augment_dim = input_dim + 1 
        
        # Define convolution layer based on type
        if conv_type == "GAT":
            Conv = GATv2Conv 
            hidden_per_head = hidden_dim // heads
        else:
            Conv = GCNConv if conv_type == "GCN" else GraphConv
            hidden_per_head = hidden_dim 
            heads = 1 

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input Layer
        if conv_type == "GAT":
            self.convs.append(Conv(self.feature_augment_dim, hidden_per_head, heads=heads, concat=True))
        else:
            self.convs.append(Conv(self.feature_augment_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden Layers
        for _ in range(num_layers):
            if conv_type == "GAT":
                self.convs.append(Conv(hidden_dim, hidden_per_head, heads=heads, concat=True))
            else:
                self.convs.append(Conv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output Layer
        if conv_type == "GAT":
            self.convs.append(Conv(hidden_dim, embedding_dim, heads=1, concat=False))
        else:
            self.convs.append(Conv(hidden_dim, embedding_dim))
        self.bns.append(nn.BatchNorm1d(embedding_dim))

        self.dropout_layer = nn.Dropout(dropout)
        self.pooling = global_add_pool 

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        # 1. Structural Feature augmentation
        deg = degree(edge_index[1], x.size(0), dtype=x.dtype).view(-1, 1)
        x = torch.cat([x, deg], dim=1)
        
        # 2. Noise injection for robust feature learning (Training only)
        if self.training:
            x = x + torch.randn_like(x) * 0.01

        # 3. Message Passing
        for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x) if self.conv_type == "GAT" else F.relu(x)
            x = self.dropout_layer(x)

        # Final Layer
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = F.leaky_relu(x, 0.2) 
        
        node_embeddings = x
        graph_embeddings = self.pooling(node_embeddings, batch)
        return node_embeddings, graph_embeddings

class F2PopulationLevelGraph(nn.Module):
    """
    Population-level graph construction module (F2).
    Learns a latent graph structure connecting patient representations.
    Regularized by Node Degree Distribution Loss (NDDL).
    """
    def __init__(self, embedding_dim: int, latent_dim: int, temperature: float = 1.0, threshold: float = 0.5):
        super().__init__()
        self.latent_transform = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        # Learnable graph construction parameters
        self.temp = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        
        # NDDL target distribution parameters
        self.mu = nn.Parameter(torch.tensor(9.0, dtype=torch.float32)) 
        self.sigma = nn.Parameter(torch.tensor(5.0, dtype=torch.float32)) 
        
    def forward(self, graph_embeddings: torch.Tensor):
        # 1. Project to latent space
        latent_space = self.latent_transform(graph_embeddings)
        
        # 2. Compute Pairwise Euclidean Distances
        diff = latent_space.unsqueeze(1) - latent_space.unsqueeze(0)
        dist_sq = torch.pow(diff, 2).sum(2)
        dist = torch.sqrt(dist_sq + 1e-6)
        
        # 3. Learn Adjacency Matrix
        logits = -self.temp * dist + self.theta
        adjacency_matrix = torch.sigmoid(logits)
        
        # Enforce self-loops and remove diagonal from learnable weights
        mask = torch.eye(adjacency_matrix.size(0), device=adjacency_matrix.device)
        adjacency_matrix = adjacency_matrix * (1 - mask) + mask 
        
        # 4. Extract Sparse Edge Index
        edge_indices = torch.nonzero(adjacency_matrix > 0.5, as_tuple=False)
        edge_index = edge_indices.t()
        edge_weight = adjacency_matrix[edge_indices[:, 0], edge_indices[:, 1]]
        
        # 5. Compute Regularization Loss
        kl_loss = self._compute_nddl(adjacency_matrix)
        return adjacency_matrix, edge_index, edge_weight, kl_loss
    
    def _compute_nddl(self, adj: torch.Tensor) -> torch.Tensor:
        """Computes Node Degree Distribution Loss (KL Divergence)."""
        N = adj.shape[0]
        device = adj.device
        
        # Differentiable Degree Approximation
        mask = (adj > 0.5).float().detach() 
        A_bar = adj * mask
        d_bar = torch.sum(A_bar, dim=1)
        
        # Soft Histogram of Degrees
        c = torch.arange(0, N, device=device).float()
        delta = d_bar.unsqueeze(1) - c.unsqueeze(0)
        epsilon = 1.0 
        S = torch.exp(- (delta ** 2) / (epsilon ** 2))
        
        # Empirical Distribution
        numerator = torch.sum(S, dim=0) 
        q = numerator / (torch.sum(numerator) + 1e-8)
        
        # Target Gaussian Distribution
        r = torch.exp(- ((c - self.mu) ** 2) / (2 * self.sigma ** 2))
        r = r / (torch.sum(r) + 1e-8)
        
        # KL Divergence
        kl = torch.sum(q * torch.log(q / (r + 1e-8) + 1e-8))
        return torch.clamp(kl, min=0, max=10.0)

class F3Classifier(nn.Module):
    """
    Population-level classifier (F3).
    Performs message passing on the learned population graph and predicts class labels.
    """
    def __init__(self, input_dim_h: int, gnn_hidden_dim: int, num_classes: int, 
                 conv_type: str = "GCN", dropout: float = 0.3, heads: int = 4, gnn_layers: int = 2):
        super().__init__()
        self.conv_type = conv_type
        self.dropout_val = dropout
        
        if conv_type == "GAT":
            Conv = GATv2Conv
            hidden_per_head = gnn_hidden_dim // heads
        else:
            Conv = GCNConv if conv_type == "GCN" else GraphConv
            hidden_per_head = gnn_hidden_dim
            heads = 1
            
        self.input_transform = nn.Sequential(
            nn.Linear(input_dim_h, gnn_hidden_dim),
            nn.LayerNorm(gnn_hidden_dim),
            nn.GELU()
        )
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            if conv_type == "GAT":
                self.gnn_layers.append(Conv(gnn_hidden_dim, hidden_per_head, heads=heads, concat=True))
            else:
                self.gnn_layers.append(Conv(gnn_hidden_dim, gnn_hidden_dim))
        
        # High-Capacity Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim * 2),
            nn.LayerNorm(gnn_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim * 2),
            nn.LayerNorm(gnn_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(gnn_hidden_dim * 2, num_classes)
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        h = self.input_transform(h)
        for gnn in self.gnn_layers:
            if self.conv_type == "GAT":
                 h = gnn(h, edge_index)
            else:
                 if edge_weight is not None:
                     h = gnn(h, edge_index, edge_weight)
                 else:
                     h = gnn(h, edge_index)
            h = F.elu(h) if self.conv_type == "GAT" else F.relu(h)
            h = F.dropout(h, p=self.dropout_val, training=self.training)
        
        return self.classifier(h)

class GiG(nn.Module):
    """
    End-to-End Graph-in-Graph (GiG) Model.
    Co-trains node-level feature extraction, latent graph learning, and semi-supervised classification.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.debug_mode = config.get("debug_mode", False)
        heads = config.get("heads", 4)
        
        self.node_level_module = F1NodeLevelModule(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            embedding_dim=config["embedding_dim"],
            conv_type=config["conv_type"],
            dropout=config["dropout"],
            num_layers=config["gnn_layers"],
            heads=heads
        )
        
        self.population_level_module = F2PopulationLevelGraph(
            embedding_dim=config["embedding_dim"],
            latent_dim=config["latent_dim"],
            threshold=0.5
        )
        
        self.classifier = F3Classifier(
            input_dim_h=config["embedding_dim"],
            gnn_hidden_dim=config["gnn_hidden_dim"],
            num_classes=config["num_classes"],
            conv_type=config["conv_type"],
            gnn_layers=config["gnn_layers"],
            dropout=config["dropout"],
            heads=heads
        )

    def forward(self, data):
        # 1. Feature Generation (F1)
        x = data.x.float()
        node_embeddings, graph_embeddings = self.node_level_module(x=x, edge_index=data.edge_index, batch=data.batch)
        
        # 2. Population Graph Learning (F2)
        adjacency_matrix, edge_index, edge_weight, kl_loss = self.population_level_module(graph_embeddings)
        
        if self.debug_mode:
            std_dev = torch.std(graph_embeddings, dim=0).mean().item()
            print(f"[DEBUG] Patient Emb StdDev: {std_dev:.5f} | Edges: {edge_index.shape[1]}")
            self.debug_mode = False

        # 3. Classification (F3)
        logits = self.classifier(h=graph_embeddings, edge_index=edge_index, edge_weight=edge_weight)
        
        return logits, adjacency_matrix, kl_loss