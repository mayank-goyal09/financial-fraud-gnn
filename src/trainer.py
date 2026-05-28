import torch
import torch.nn.functional as F

def train_model(model, data, epochs=100):
    # 1. Optimizer: Adam is the industry standard for GNNs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 2. Handle Class Imbalance
    # We tell the model: "Fraud (1) is 10x more important than Legitimate (0)"
    weights = torch.tensor([1.0, 10.0]) 
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad() # Reset gradients
        
        # Forward pass: Model makes a guess
        out = model(data)
        
        # Calculate Loss using our weights
        loss = F.nll_loss(out, data.y, weight=weights)
        
        # Backward pass: Model learns from mistakes
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    return model

def get_fraud_scores(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        # Convert logits to probabilities
        probs = torch.exp(logits) 
        # The second column is the probability of class 1 (Fraud)
        fraud_probabilities = probs[:, 1]
    return fraud_probabilities
# --- RUNNING THE TRAIN ---
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model import FraudGCN
    
    print("Loading processed graph...")
    loaded_graph = torch.load("data/processed_graph.pt", weights_only=False)
    
    # We dynamically get input_dim from the node features shape
    print("Initializing Model...")
    model = FraudGCN(input_dim=loaded_graph.x.shape[1], hidden_dim=16, output_dim=2)
    
    print("Training Model...")
    trained_model = train_model(model, loaded_graph)
    
    # Save the model
    torch.save(trained_model.state_dict(), "data/trained_model.pt")
    print("\nModel trained and saved successfully to data/trained_model.pt")