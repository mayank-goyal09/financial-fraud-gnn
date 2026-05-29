import os
import sys
import torch

# Ensure the root directory is on the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.simulator import generate_bank_data
from src.data_processor import process_data
from src.model import FraudGCN
from src.trainer import train_model, get_fraud_scores
from src.visualizer import extract_and_plot_suspect

def main():
    print("=" * 60)
    print("      FINANCIAL FRAUD DETECTION USING GRAPH NEURAL NETWORKS      ")
    print("=" * 60)
    
    # Paths
    raw_data_path = "data/raw_transactions.csv"
    processed_graph_path = "data/processed_graph.pt"
    model_path = "data/trained_model.pt"
    
    # Step 1: Simulate data if missing
    if not os.path.exists(raw_data_path):
        print("\n[Step 1] Raw data not found. Simulating transaction network...")
        generate_bank_data(num_users=1000, num_normal_tx=5000)
    else:
        print("\n[Step 1] Found existing raw transaction data.")
        
    # Step 2: Process transaction data into a graph
    print("\n[Step 2] Preprocessing transactions into Graph structure...")
    graph = process_data(raw_data_path, processed_graph_path)
    
    # Step 3: Train Graph Neural Network (GNN)
    print("\n[Step 3] Initializing and training GCN Model...")
    input_dim = graph.x.shape[1]
    model = FraudGCN(input_dim=input_dim, hidden_dim=16, output_dim=2)
    trained_model, losses = train_model(model, graph, epochs=100)
    
    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model checkpoint saved to {model_path}")
    
    # Step 4: Evaluate and Identify Suspects
    print("\n[Step 4] Predicting risk scores for all accounts...")
    scores = get_fraud_scores(trained_model, graph)
    
    top_suspect = torch.argmax(scores).item()
    print(f"\nTargeting Suspect Node ID: {top_suspect} with Score: {scores[top_suspect].item():.4f}")
    
    # Step 5: Ego-Graph Visualization
    print("\n[Step 5] Isolating ego-graph neighborhood of top suspect...")
    extract_and_plot_suspect(graph, suspect_id=top_suspect, scores=scores, hops=2)
    
    print("\n" + "=" * 60)
    print("                  PIPELINE RUN COMPLETED                     ")
    print("=" * 60)

if __name__ == "__main__":
    main()
