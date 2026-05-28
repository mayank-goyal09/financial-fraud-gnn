import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import k_hop_subgraph
import os
import sys

def extract_and_plot_suspect(data, suspect_id, scores, hops=2):
    print(f"\n[INVESTIGATION] Analyzing N-Hop Ecosystem around Node {suspect_id}...")
    # 1. Use PyG utility to find all nodes within N hops of the suspect
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        suspect_id, hops, data.edge_index, relabel_nodes=True, directed=False
    )
    
    # 2. Convert this small "ego-graph" to NetworkX
    G = nx.DiGraph()
    
    # Add edges for the subgraph
    # We need to map the relabeled nodes back to their original IDs for the labels
    node_list = subset.tolist()
    
    # Extract edge attributes if they exist
    has_edge_attr = hasattr(data, 'edge_attr') and data.edge_attr is not None
    
    # The mask tells us which edges out of the ENTIRE graph belong to this sub-graph
    original_edge_indices = edge_mask.nonzero(as_tuple=True)[0].tolist()
    
    # For every edge in our localized subgraph, map it to the original list to grab the amount transferred!
    for i in range(edge_index.shape[1]):
        src = node_list[edge_index[0, i]]
        dst = node_list[edge_index[1, i]]
        
        # Grab the original edge amount weight using the mask
        if has_edge_attr:
            orig_edge_idx = original_edge_indices[i]
            amount = data.edge_attr[orig_edge_idx].item()
        else:
            amount = 1.0
            
        G.add_edge(src, dst, weight=amount)
        
    # 3. Visualization configuration
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Color nodes by their probability score (Red = High, Green = Low)
    node_colors = [scores[node].item() for node in G.nodes()]
    
    # Dynamic Edge Thickness based on amount
    edges_list = G.edges(data=True)
    if not edges_list:
        print("No connections found to plot.")
        return
        
    weights = [attr['weight'] for u, v, attr in edges_list]
    max_w = max(weights) if weights and max(weights) > 0 else 1
    edge_thickness = [1.0 + 5.0 * (w / max_w) for w in weights]
    
    print(f"Creating highly-focused graph mapping {len(list(G.nodes))} entities...")
    
    # Draw Nodes
    nodes = nx.draw_networkx_nodes(G, pos, 
                                   node_color=node_colors, 
                                   cmap=plt.cm.RdYlGn_r, 
                                   vmin=0.0, vmax=1.0,
                                   node_size=800, edgecolors='black')
                                   
    # Draw Edges (Arrows)                      
    nx.draw_networkx_edges(G, pos, 
                           width=edge_thickness, 
                           edge_color='gray', 
                           arrows=True, 
                           arrowsize=20, 
                           connectionstyle='arc3,rad=0.1')
                           
    # Draw Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_weight='bold')
    
    # Add a colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar.set_label('Fraud Probability Score', rotation=270, labelpad=20)
    
    plt.title(f"Fraud Investigation Report\nNeighborhood Isolation of Top Suspect: Node {suspect_id}\n(Network Distance: {hops}-Hops)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model import FraudGCN
    from src.trainer import get_fraud_scores
    
    # Load graph
    print("Loading graph data...")
    graph = torch.load("data/processed_graph.pt", weights_only=False)
    
    # Init and load model
    print("Loading trained model...")
    model = FraudGCN(input_dim=graph.x.shape[1], hidden_dim=16, output_dim=2)
    model.load_state_dict(torch.load("data/trained_model.pt", weights_only=True))
    
    print("Generating fraud scores...")
    scores = get_fraud_scores(model, graph)
    max_score = scores.max().item()
    print(f"Scores calculated! Max score found: {max_score:.4f}")
    
    # --- ADD CLASSIFICATION REPORT EVALUATION ---
    print("\n[VALIDATION] Generating Classification Report...")
    from sklearn.metrics import classification_report
    
    eval_threshold = 0.5 
    y_true = graph.y.numpy()
    y_pred = (scores > eval_threshold).numpy()
    
    if len(set(y_true)) > 1: # Only run classification report if both classes actually exist
        print(f"Threshold Set: > {eval_threshold}")
        print(classification_report(y_true, y_pred, target_names=["Legitimate (0)", "Fraud (1)"]))
    else:
        print("Data holds only a single class; skipping classification report.")
    
    # Identify the Top Suspect Node
    # Torch argmax inherently finds the index of the highest probability value
    top_suspect = torch.argmax(scores).item()
    
    print(f"\nTargeting Suspect Node ID: {top_suspect} with Score: {scores[top_suspect].item():.4f}")
    print("Extracting localized network evidence...")
    
    # Plot only the relevant Evidence Ring (Ego-Graph)
    extract_and_plot_suspect(graph, suspect_id=top_suspect, scores=scores, hops=2)