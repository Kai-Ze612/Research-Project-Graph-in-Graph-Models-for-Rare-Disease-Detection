import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GIN, global_mean_pool, global_add_pool

class GlobalNodeEmbedding(nn.Module):
    def __init__(self, num_global_nodes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_global_nodes, embedding_dim)
   
    def forward(self, node_ids):
        if isinstance(node_ids, list):
            node_ids = torch.tensor(node_ids, dtype=torch.long)
        node_ids = node_ids.view(-1)
        return self.embedding(node_ids)

class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super().__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = (max - min) / bins
        self.centers = torch.linspace(min + self.delta/2, max - self.delta/2, bins)
        
    def forward(self, x):
        x = x.view(-1, 1)
        centers = self.centers.to(x.device).view(1, -1)
        x = torch.exp(-(x - centers)**2 / (2 * self.sigma**2))
        x = x / x.sum(dim=1, keepdim=True)
        histogram = x.sum(dim=0)
        return histogram

class F1NodeLevelModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, conv_type='GCN',
                 dropout=0.5, pooling="mean", num_layers=2):
        super().__init__()
        self.dropout = dropout

        if conv_type == "GCN": Conv = GCNConv
        elif conv_type == "Graph": Conv = GraphConv
        elif conv_type == "SAGE": Conv = SAGEConv
        else: raise ValueError(f"Unknown conv_type: {conv_type}")

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(Conv(input_dim, hidden_dim))
        self.bns.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers):
            self.convs.append(Conv(hidden_dim, hidden_dim))
            self.bns.append(nn.LayerNorm(hidden_dim))

        self.convs.append(Conv(hidden_dim, embedding_dim))
        self.bns.append(nn.LayerNorm(embedding_dim))

        self.dropout_layer = nn.Dropout(dropout)
        self.pooling = global_mean_pool if pooling == "mean" else global_add_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout_layer(x)

        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout_layer(x)

        node_embeddings = x
        graph_embeddings = self.pooling(node_embeddings, batch)
        return node_embeddings, graph_embeddings

class F2PopulationLevelGraph(nn.Module):
    def __init__(self, embedding_dim, latent_dim, temperature=0.5, threshold=0.1):
        super().__init__()
        self.latent_transform = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.2)
        )
        self.temp = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.mu = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
    def forward(self, graph_embeddings):
        latent_space = self.latent_transform(graph_embeddings)
        diff = latent_space.unsqueeze(1) - latent_space.unsqueeze(0)
        diff = torch.pow(diff, 2).sum(2)
        mask_diff = diff != 0.0
        dist = - torch.sqrt(diff + torch.finfo(torch.float32).eps)
        dist = dist * mask_diff
        
        prob_matrix = self.temp * dist + self.theta
        adj = prob_matrix + torch.eye(prob_matrix.shape[0]).to(prob_matrix.device)
        adjacency_matrix = torch.sigmoid(adj)
        
        edge_indices = torch.nonzero(adjacency_matrix > 0.1, as_tuple=False)
        edge_index = edge_indices.t()
        edge_weight = adjacency_matrix[edge_indices[:, 0], edge_indices[:, 1]]
        
        n_nodes = adjacency_matrix.shape[0]
        softhist = SoftHistogram(bins=n_nodes, min=0.5, max=n_nodes + 0.5, sigma=0.6)
        kl_loss = self._compute_kl_loss(adjacency_matrix, n_nodes, softhist)
        
        return adjacency_matrix, edge_index, edge_weight, kl_loss
    
    def _compute_kl_loss(self, adj, batch_size, softhist):
        binarized_adj = torch.zeros(adj.shape).to(adj.device)
        binarized_adj[adj > 0.5] = 1
        dist, deg = self._compute_distr(adj * binarized_adj, softhist)
        target_dist = self._compute_target_distribution(batch_size)
        return torch.sum(dist * torch.log(dist / (target_dist + 1e-8) + 1e-8))
    
    def _compute_distr(self, adj, softhist):
        deg = adj.sum(-1)
        distr = softhist(deg)
        return distr / torch.sum(distr), deg
    
    def _compute_target_distribution(self, batch_size):
        device = self.mu.device
        indices = torch.arange(batch_size, device=device)
        target_distribution = torch.exp(-((self.mu - indices) ** 2) / (self.sigma ** 2))
        return target_distribution / target_distribution.sum()

class F3Classifier(nn.Module):
    def __init__(self, input_dim_h, gnn_hidden_dim, num_classes, conv_type="GCN", gnn_layers=2, dropout=0.3):
        super().__init__()
        self.dropout_val = dropout
        Conv = GCNConv if conv_type == "GCN" else GraphConv
        
        self.input_transform = nn.Sequential(
            nn.Linear(input_dim_h, gnn_hidden_dim),
            nn.BatchNorm1d(gnn_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        self.gnn_layers = nn.ModuleList([Conv(gnn_hidden_dim, gnn_hidden_dim) for _ in range(gnn_layers)])
        
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.BatchNorm1d(gnn_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, num_classes)
        )

    def forward(self, h, edge_index, edge_weight=None):
        h = self.input_transform(h)
        for gnn in self.gnn_layers:
            if edge_weight is not None and isinstance(gnn, (GCNConv, GraphConv)):
                h = gnn(h, edge_index, edge_weight)
            else:
                h = gnn(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_val, training=self.training)
        
        logits = self.classifier(h)
        return logits

class GiG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.node_level_module = F1NodeLevelModule(
            input_dim=config["input_dim"], hidden_dim=config["hidden_dim"],
            embedding_dim=config["embedding_dim"], conv_type=config["conv_type"], dropout=config["dropout"]
        )
        self.population_level_module = F2PopulationLevelGraph(
            embedding_dim=config["embedding_dim"], latent_dim=config["latent_dim"]
        )
        self.classifier = F3Classifier(
            input_dim_h=config["embedding_dim"], gnn_hidden_dim=config["gnn_hidden_dim"],
            num_classes=config["num_classes"], conv_type=config["conv_type"],
            gnn_layers=config["gnn_layers"], dropout=config["dropout"]
        )

    def forward(self, data):
        node_embeddings, graph_embeddings = self.node_level_module(data)
        adjacency_matrix, edge_index, edge_weight, kl_loss = self.population_level_module(graph_embeddings)
        logits = self.classifier(h=graph_embeddings, edge_index=edge_index, edge_weight=edge_weight)
        return logits, adjacency_matrix, kl_loss