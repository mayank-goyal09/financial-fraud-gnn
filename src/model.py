import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FraudGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FraudGCN, self).__init__()
        # Layer 1: Takes raw features and compresses them into a "hidden" representation
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
        # Layer 2: Refines that representation to make a final prediction
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 1. First Message Passing
        x = self.conv1(x, edge_index)
        x = F.relu(x) # Activation: "Ignite" the important signals
        x = F.dropout(x, p=0.5, training=self.training) # Prevent overfitting

        # 2. Second Message Passing
        x = self.conv2(x, edge_index)

        # Output: Log-Softmax for classification (Fraud vs. Not Fraud)
        return F.log_softmax(x, dim=1)