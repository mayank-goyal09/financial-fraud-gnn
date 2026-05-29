# Financial Fraud Detection using Graph Neural Networks (GNNs)

A state-of-the-art machine learning pipeline utilizing Graph Convolutional Networks (GCNs) to detect complex financial fraud patterns (such as money laundering smurfing rings, criminal out-flows, and high-volume collectors) within simulated banking transaction networks.

## 🌟 Key Features
- **Bank Network Simulation**: Artificially synthesizes high-fidelity banking transactions containing normal patterns alongside injection of specific fraudulent activities:
  - **Criminal Outflow**: High out-degree entities distributing assets.
  - **Collector Inflow**: High in-degree hubs gathering assets.
  - **Smurfing Ring**: Multi-layered money-laundering topologies (1 Boss -> 100 Smurfs -> 1 Hub).
- **Heterogeneous Graph Construction**: Compiles raw logs into structured graph representations (Users and Transactions as nodes, connected via weighted transaction flows).
- **Deep Graph Representation Learning**: Uses a 2-layer Graph Convolutional Network (GCN) built on PyTorch Geometric to perform node classification.
- **Explainable AI Neighborhood Investigation**: Isolates and visualizes the N-hop ego-graph around detected high-risk fraud suspects.

## 🛠️ Project Structure
```text
project 64 fraud detection/
├── data/
│   ├── raw_transactions.csv   # Synthesized transaction log
│   ├── processed_graph.pt     # Serialized PyTorch Geometric Graph
│   └── trained_model.pt       # Trained GNN weights
├── notebooks/
│   └── main.ipynb             # Interactive analysis playground
├── src/
│   ├── data_processor.py      # Cleans & maps transaction logs to a Graph
│   ├── model.py               # FraudGCN PyTorch module
│   ├── trainer.py             # Optimizer setup, loss weighting & training
│   └── visualizer.py          # NetworkX ego-graph neighborhood plotter
├── main.py                    # End-to-end pipeline orchestrator
└── requirements.txt           # Environment dependencies
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
Simply execute the main orchestration script to run simulation, processing, model training, evaluation, and visualization:
```bash
python main.py
```

### 3. Interactive Visualization
Open `notebooks/main.ipynb` to customize the visualization parameters and run interactive ad-hoc investigations.

## 📊 GNN Model Architecture
The custom `FraudGCN` model implements a robust Graph Convolutional Network topology:
- **Layer 1**: Graph Convolution (`GCNConv`) compressing feature dimensions to a 16-dimensional embedding space, followed by `ReLU` activation and a dropout probability of `0.5` to minimize overfitting.
- **Layer 2**: Secondary Graph Convolution mapping representation into 2-class predictions (Legitimate vs. Fraudulent).
- **Weighted Loss**: An optimized Negative Log-Likelihood loss (`NLLLoss`) using a weighted vector to tackle massive class imbalance (Fraud is weighted 10x relative to Legitimate).
