import os
import pandas as pd
import torch
from data.data import build_financial_graph

def process_data(csv_path="data/raw_transactions.csv", output_path="data/processed_graph.pt"):
    """
    Reads the raw transaction CSV, builds a financial graph, and saves it.
    """
    print(f"Reading raw transactions from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw transaction file not found at {csv_path}. Please run the simulator first.")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} transactions.")
    
    # Build graph using the builder
    graph = build_financial_graph(df)
    
    # Save the processed graph
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graph, output_path)
    print(f"Processed graph saved successfully to {output_path}")
    return graph

if __name__ == "__main__":
    # Test the processor
    process_data()
