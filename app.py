import os
import sys
import torch
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Ensure root dir is on path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.simulator import generate_bank_data
from src.data_processor import process_data
from src.model import FraudGCN
from src.trainer import train_model, get_fraud_scores

app = Flask(__name__, static_folder="web")
CORS(app)

RAW_DATA_PATH = "data/raw_transactions.csv"
PROCESSED_GRAPH_PATH = "data/processed_graph.pt"
MODEL_PATH = "data/trained_model.pt"

# Helper: reconstruct node name mapping from CSV
def get_node_names_and_types():
    if not os.path.exists(RAW_DATA_PATH):
        return [], 0
    df = pd.read_csv(RAW_DATA_PATH)
    users = list(pd.concat([df['sender'], df['receiver']]).unique())
    transactions = list(df['tx_id'].unique())
    all_nodes = users + transactions
    return all_nodes, len(users)

def resolve_node_role(name):
    if name.startswith("U_CRIM_"):
        return "Criminal Outflow"
    elif name.startswith("U_COLL_"):
        return "Collector Inflow"
    elif name.startswith("U_NORM_"):
        return "Normal User"
    elif name.startswith("U_SMURF_"):
        return "Smurf"
    elif name == "U_FRAUD_BOSS":
        return "Fraud Boss"
    elif name == "U_FRAUD_HUB":
        return "Fraud Hub"
    elif name.startswith("TX_"):
        return "Transaction"
    return "Unknown"

# Server Frontend Static Routes
@app.route("/")
def serve_index():
    return send_from_directory("web", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("web", path)

# REST API Endpoints
@app.route("/api/status", methods=["GET"])
def get_status():
    status = {
        "simulation_exists": os.path.exists(RAW_DATA_PATH),
        "graph_processed": os.path.exists(PROCESSED_GRAPH_PATH),
        "model_trained": os.path.exists(MODEL_PATH),
        "simulation_stats": None,
        "graph_stats": None
    }
    
    if status["simulation_exists"]:
        try:
            df = pd.read_csv(RAW_DATA_PATH)
            status["simulation_stats"] = {
                "total_transactions": len(df),
                "fraud_transactions": int(df['is_fraud'].sum()),
                "legit_transactions": int(len(df) - df['is_fraud'].sum()),
                "total_users": int(pd.concat([df['sender'], df['receiver']]).nunique())
            }
        except Exception as e:
            print("Error reading CSV stats:", e)
            
    if status["graph_processed"]:
        try:
            graph = torch.load(PROCESSED_GRAPH_PATH, weights_only=False)
            status["graph_stats"] = {
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
                "num_features": graph.x.shape[1]
            }
        except Exception as e:
            print("Error loading graph stats:", e)
            
    return jsonify(status)

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json() or {}
    num_users = int(data.get("num_users", 1000))
    num_normal_tx = int(data.get("num_normal_tx", 5000))
    
    print(f"Simulating network with users={num_users}, normal_transactions={num_normal_tx}")
    
    try:
        df = generate_bank_data(num_users=num_users, num_normal_tx=num_normal_tx)
        
        # Calculate dynamic info
        total_tx = len(df)
        fraud_tx = int(df['is_fraud'].sum())
        legit_tx = total_tx - fraud_tx
        total_users = int(pd.concat([df['sender'], df['receiver']]).nunique())
        
        return jsonify({
            "success": True,
            "message": "Simulation successful!",
            "stats": {
                "total_transactions": total_tx,
                "fraud_transactions": fraud_tx,
                "legit_transactions": legit_tx,
                "total_users": total_users
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/process", methods=["POST"])
def api_process():
    try:
        print("Preprocessing CSV raw data to Graph...")
        graph = process_data(RAW_DATA_PATH, PROCESSED_GRAPH_PATH)
        
        y_np = graph.y.numpy()
        num_fraud = int(np.sum(y_np == 1))
        num_legit = int(np.sum(y_np == 0))
        
        return jsonify({
            "success": True,
            "message": "Data successfully processed into Heterogeneous Financial Graph!",
            "stats": {
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
                "num_fraud_nodes": num_fraud,
                "num_legit_nodes": num_legit
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def api_train():
    data = request.get_json() or {}
    epochs = int(data.get("epochs", 100))
    lr = float(data.get("lr", 0.01))
    
    if not os.path.exists(PROCESSED_GRAPH_PATH):
        return jsonify({"success": False, "error": "Processed graph not found. Please preprocess data first."}), 400
        
    try:
        print(f"Loading graph and starting GNN training for {epochs} epochs...")
        graph = torch.load(PROCESSED_GRAPH_PATH, weights_only=False)
        
        # Init model
        input_dim = graph.x.shape[1]
        model = FraudGCN(input_dim=input_dim, hidden_dim=16, output_dim=2)
        
        # Train
        trained_model, losses = train_model(model, graph, epochs=epochs, lr=lr)
        
        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(trained_model.state_dict(), MODEL_PATH)
        
        # Calculate classification report on the whole graph
        scores = get_fraud_scores(trained_model, graph)
        y_true = graph.y.numpy()
        y_pred = (scores > 0.5).numpy().astype(int)
        
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        return jsonify({
            "success": True,
            "losses": losses,
            "metrics": {
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4)
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/suspects", methods=["GET"])
def api_suspects():
    if not os.path.exists(PROCESSED_GRAPH_PATH) or not os.path.exists(MODEL_PATH):
        return jsonify({"success": False, "error": "Graph or trained model weights missing. Please process and train first."}), 400
        
    try:
        graph = torch.load(PROCESSED_GRAPH_PATH, weights_only=False)
        
        input_dim = graph.x.shape[1]
        model = FraudGCN(input_dim=input_dim, hidden_dim=16, output_dim=2)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        
        scores = get_fraud_scores(model, graph)
        all_nodes, num_users = get_node_names_and_types()
        
        # Build suspects list
        suspects = []
        for idx in range(graph.num_nodes):
            name = all_nodes[idx] if idx < len(all_nodes) else f"Node_{idx}"
            role = resolve_node_role(name)
            is_user = idx < num_users
            
            # Extract features for this node
            # x format: [Amount_val, is_user, is_transaction, hour_of_day]
            features = graph.x[idx].numpy()
            amount_val = float(features[0])
            hour_val = int(features[3])
            
            score_val = float(scores[idx].item())
            true_label = int(graph.y[idx].item())
            
            suspects.append({
                "id": idx,
                "name": name,
                "role": role,
                "is_user": is_user,
                "avg_amount": round(amount_val, 2),
                "hour": hour_val,
                "score": round(score_val, 4),
                "actual_label": true_label
            })
            
        # Sort suspects by threat score descending
        suspects = sorted(suspects, key=lambda x: x["score"], reverse=True)
        
        return jsonify({
            "success": True,
            "suspects": suspects
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/investigate/<int:suspect_id>", methods=["GET"])
def api_investigate(suspect_id):
    hops = int(request.args.get("hops", 2))
    
    if not os.path.exists(PROCESSED_GRAPH_PATH) or not os.path.exists(MODEL_PATH):
        return jsonify({"success": False, "error": "Trained data missing."}), 400
        
    try:
        graph = torch.load(PROCESSED_GRAPH_PATH, weights_only=False)
        
        input_dim = graph.x.shape[1]
        model = FraudGCN(input_dim=input_dim, hidden_dim=16, output_dim=2)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        
        scores = get_fraud_scores(model, graph)
        all_nodes, num_users = get_node_names_and_types()
        
        from torch_geometric.utils import k_hop_subgraph
        
        # Extract N-Hop sub-graph
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            suspect_id, hops, graph.edge_index, relabel_nodes=True, directed=False
        )
        
        node_list = subset.tolist()
        
        # Build Nodes JSON list
        nodes_json = []
        for local_id, global_id in enumerate(node_list):
            name = all_nodes[global_id] if global_id < len(all_nodes) else f"Node_{global_id}"
            role = resolve_node_role(name)
            features = graph.x[global_id].numpy()
            amount_val = float(features[0])
            
            nodes_json.append({
                "id": global_id,
                "label": name,
                "role": role,
                "is_user": global_id < num_users,
                "score": round(float(scores[global_id].item()), 4),
                "avg_amount": round(amount_val, 2),
                "is_target": global_id == suspect_id
            })
            
        # Build Edges JSON list
        edges_json = []
        original_edge_indices = edge_mask.nonzero(as_tuple=True)[0].tolist()
        
        has_edge_attr = hasattr(graph, 'edge_attr') and graph.edge_attr is not None
        
        for i in range(edge_index.shape[1]):
            src = node_list[edge_index[0, i]]
            dst = node_list[edge_index[1, i]]
            
            if has_edge_attr:
                orig_edge_idx = original_edge_indices[i]
                amount = float(graph.edge_attr[orig_edge_idx].item())
            else:
                amount = 1.0
                
            edges_json.append({
                "from": src,
                "to": dst,
                "amount": round(amount, 2)
            })
            
        return jsonify({
            "success": True,
            "suspect_id": suspect_id,
            "hops": hops,
            "nodes": nodes_json,
            "edges": edges_json
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("=" * 60)
    print("     STARTING FRAUD DETECTION GRAPH DASHBOARD SERVER        ")
    print("=" * 60)
    print("Running local web server at http://127.0.0.1:8085")
    app.run(debug=True, use_reloader=False, port=8085)
