import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import os

def build_financial_graph(df):
    print("Building Heterogeneous financial graph (User -> Transaction -> User)...")
    
    # 1. Get unique nodes
    users = list(pd.concat([df['sender'], df['receiver']]).unique())
    transactions = list(df['tx_id'].unique())
    
    all_nodes = users + transactions
    mapping = {node: i for i, node in enumerate(all_nodes)}
    
    # 2. Create the Edge Index
    # Sender -> Transaction
    s2t_source = df['sender'].map(mapping).values
    s2t_target = df['tx_id'].map(mapping).values
    
    # Transaction -> Receiver
    t2r_source = df['tx_id'].map(mapping).values
    t2r_target = df['receiver'].map(mapping).values
    
    source_nodes = np.concatenate([s2t_source, t2r_source])
    target_nodes = np.concatenate([s2t_target, t2r_target])
    edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long)
    
    # Edge Attributes (amount goes on the edges to the transaction for thickness)
    s2t_amounts = df['amount'].values
    t2r_amounts = df['amount'].values
    edge_attr = torch.tensor(np.concatenate([s2t_amounts, t2r_amounts]), dtype=torch.float)
    
    # 3. Create Node Features (X)
    print("Creating Node Features...")
    # Format: [Amount_val, is_user, is_transaction, hour_of_day]
    sender_avg = df.groupby('sender')['amount'].mean()
    
    node_features = []
    # Add User Features
    for u in users:
        avg_v = sender_avg.get(u, 0.0)
        node_features.append([avg_v, 1.0, 0.0, 12.0]) # default hour for users
        
    # Add Transaction Features
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    else:
        df['hour'] = 12
        
    tx_features = df.set_index('tx_id')[['amount', 'hour']].to_dict('index')
    for t in transactions:
        tx_data = tx_features[t]
        node_features.append([tx_data['amount'], 0.0, 1.0, float(tx_data['hour'])])
        
    x = torch.tensor(node_features, dtype=torch.float)
    
    # 4. Labels (Y) - 1 for Fraud, 0 for Legitimate
    print("Creating Labels...")
    fraud_from_sender = df.groupby('sender')['is_fraud'].max()
    fraud_from_receiver = df.groupby('receiver')['is_fraud'].max()
    
    labels = []
    # For Users
    for u in users:
        f_s = fraud_from_sender.get(u, 0)
        f_r = fraud_from_receiver.get(u, 0)
        labels.append(max(f_s, f_r))
        
    # For Transactions
    tx_fraud = df.set_index('tx_id')['is_fraud'].to_dict()
    for t in transactions:
        labels.append(tx_fraud[t])
        
    y = torch.tensor(labels, dtype=torch.long)
    
    graph_data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
    return graph_data

if __name__ == "__main__":
    file_path = "data/raw_transactions.csv"
    
    # If the file exists and has data, read it. Else create dummy data for testing.
    if os.path.exists(file_path) and os.path.getsize(file_path) > 10:
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
    else:
        print("CSV is empty or missing! Generating dummy data for testing...")
        df = pd.DataFrame({
            'sender': ['A', 'B', 'B', 'C', 'A'],
            'receiver': ['B', 'C', 'D', 'D', 'C'],
            'amount': [100.0, 50.0, 200.0, 30.0, 500.0],
            'is_fraud': [0, 0, 1, 0, 1] 
        })
        
    graph = build_financial_graph(df)
    print("\n=== Graph Created Successfully ===")
    print(graph)
    print(f"Nodes (Users): {graph.num_nodes}")
    print(f"Edges (Transactions): {graph.num_edges}")
    
    # Save the processed graph for later use
    out_file = "data/processed_graph.pt"
    torch.save(graph, out_file)
    print(f"Saved graph to {out_file}")