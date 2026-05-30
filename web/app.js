// AegisGNN - Core JavaScript Orchestrator

const API_BASE = window.location.origin + "/api";

// DOM Cache
const dom = {
    usersInput: document.getElementById("input-users"),
    valUsers: document.getElementById("val-users"),
    txInput: document.getElementById("input-tx"),
    valTx: document.getElementById("val-tx"),
    
    btnRunSim: document.getElementById("btn-run-sim"),
    btnProcessGraph: document.getElementById("btn-process-graph"),
    btnTrainModel: document.getElementById("btn-train-model"),
    
    epochsInput: document.getElementById("input-epochs"),
    lrInput: document.getElementById("input-lr"),
    
    trainingProgress: document.getElementById("training-progress-text"),
    metricAcc: document.getElementById("metric-acc"),
    metricPrec: document.getElementById("metric-prec"),
    metricRec: document.getElementById("metric-rec"),
    metricF1: document.getElementById("metric-f1"),
    
    suspectSearch: document.getElementById("suspect-search"),
    suspectsTableBody: document.querySelector("#suspects-table tbody"),
    filterBtns: document.querySelectorAll(".filter-btn"),
    
    investigateMetaPane: document.getElementById("investigate-meta-pane"),
    inputHops: document.getElementById("input-hops"),
    
    clearConsoleBtn: document.getElementById("clear-console"),
    consoleLogs: document.getElementById("console-logs"),
    
    simIndicator: document.getElementById("sim-indicator"),
    graphIndicator: document.getElementById("graph-indicator"),
    modelIndicator: document.getElementById("model-indicator"),
    
    statusSim: document.getElementById("status-simulation"),
    statusGraph: document.getElementById("status-graph"),
    statusModel: document.getElementById("status-model")
};

// State Variables
let lossChart = null;
let networkEngine = null;
let allSuspects = [];
let activeSuspectId = null;
let activeFilter = "all";

// Setup dynamic slider updates
dom.usersInput.addEventListener("input", (e) => {
    dom.valUsers.textContent = parseInt(e.target.value).toLocaleString();
});
dom.txInput.addEventListener("input", (e) => {
    dom.valTx.textContent = parseInt(e.target.value).toLocaleString();
});

// Setup console logging
function log(msg, type = "info") {
    const time = new Date().toLocaleTimeString();
    const line = document.createElement("div");
    line.className = `console-line ${type}`;
    line.textContent = `[${time}] [${type.toUpperCase()}] ${msg}`;
    dom.consoleLogs.appendChild(line);
    dom.consoleLogs.scrollTop = dom.consoleLogs.scrollHeight;
}

dom.clearConsoleBtn.addEventListener("click", () => {
    dom.consoleLogs.innerHTML = "";
    log("Console cleared.", "system");
});

// Fetch system status on load to sync UI state
async function checkSystemStatus() {
    try {
        const res = await fetch(`${API_BASE}/status`);
        const status = await res.json();
        
        // Update simulation status
        if (status.simulation_exists) {
            dom.simIndicator.textContent = "Simulated";
            dom.statusSim.classList.add("active");
            dom.btnProcessGraph.removeAttribute("disabled");
            
            const stats = status.simulation_stats;
            log(`Found existing bank simulation data: ${stats.total_users.toLocaleString()} users, ${stats.total_transactions.toLocaleString()} transactions (${stats.fraud_transactions} fraud injects).`, "success");
        } else {
            dom.simIndicator.textContent = "None";
            dom.statusSim.classList.remove("active");
            dom.btnProcessGraph.setAttribute("disabled", "true");
        }
        
        // Update graph status
        if (status.graph_processed) {
            dom.graphIndicator.textContent = "Compiled";
            dom.statusGraph.classList.add("active");
            dom.btnTrainModel.removeAttribute("disabled");
            
            const stats = status.graph_stats;
            log(`Found processed PyG graph: ${stats.num_nodes.toLocaleString()} nodes, ${stats.num_edges.toLocaleString()} edges. Ready for model training.`, "success");
        } else {
            dom.graphIndicator.textContent = "Unprocessed";
            dom.statusGraph.classList.remove("active");
            dom.btnTrainModel.setAttribute("disabled", "true");
        }
        
        // Update model status
        if (status.model_trained) {
            dom.modelIndicator.textContent = "Trained";
            dom.statusModel.classList.add("active");
            log(`Found existing FraudGCN neural weights. Risk directory unlocked.`, "success");
            
            // Auto-load suspects directory
            loadSuspectsDirectory();
        } else {
            dom.modelIndicator.textContent = "Untrained";
            dom.statusModel.classList.remove("active");
        }
        
    } catch (e) {
        log(`Failed to contact AegisGNN backend server: ${e.message}. Please check if app.py is running.`, "error");
    }
}

// 1. RUN SIMULATION
dom.btnRunSim.addEventListener("click", async () => {
    const numUsers = dom.usersInput.value;
    const numTx = dom.txInput.value;
    
    setButtonLoading(dom.btnRunSim, true, "Simulating Bank...");
    log(`Initializing bank transaction network simulation (Users: ${numUsers}, Inbound Transactions: ${numTx})...`, "info");
    
    try {
        const res = await fetch(`${API_BASE}/simulate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ num_users: numUsers, num_normal_tx: numTx })
        });
        const result = await res.json();
        
        if (result.success) {
            log(`Bank simulation completed successfully! Created ${result.stats.total_transactions.toLocaleString()} transaction edges linking ${result.stats.total_users.toLocaleString()} normal/fraud nodes.`, "success");
            checkSystemStatus();
        } else {
            log(`Simulator crash: ${result.error}`, "error");
        }
    } catch (e) {
        log(`API Communication error: ${e.message}`, "error");
    } finally {
        setButtonLoading(dom.btnRunSim, false, "Run Simulation");
    }
});

// 2. PROCESS DATA
dom.btnProcessGraph.addEventListener("click", async () => {
    setButtonLoading(dom.btnProcessGraph, true, "Preprocessing Graph...");
    log("Compressing raw transaction logs into custom PyTorch Geometric multi-dimensional graph tensor...", "info");
    
    try {
        const res = await fetch(`${API_BASE}/process`, { method: "POST" });
        const result = await res.json();
        
        if (result.success) {
            const stats = result.stats;
            log(`Graph compiled successfully! Constructed Heterogeneous Network containing ${stats.num_nodes.toLocaleString()} node features (${stats.num_fraud_nodes} labeled anomaly points, ${stats.num_legit_nodes} normal points) and ${stats.num_edges.toLocaleString()} edges.`, "success");
            checkSystemStatus();
        } else {
            log(`Processor crash: ${result.error}`, "error");
        }
    } catch (e) {
        log(`API Communication error: ${e.message}`, "error");
    } finally {
        setButtonLoading(dom.btnProcessGraph, false, "Build Neural Graph");
    }
});

// 3. TRAIN GNN MODEL
dom.btnTrainModel.addEventListener("click", async () => {
    const epochs = dom.epochsInput.value;
    const lr = dom.lrInput.value;
    
    setButtonLoading(dom.btnTrainModel, true, "Training GCN...");
    dom.trainingProgress.textContent = "TRAINING...";
    dom.trainingProgress.classList.add("training");
    log(`Initializing FraudGCN optimization (Epochs: ${epochs}, Learning Rate: ${lr}, weighted NLLLoss)...`, "info");
    
    // Clear old chart
    if (lossChart) {
        lossChart.destroy();
        lossChart = null;
    }
    
    try {
        const res = await fetch(`${API_BASE}/train`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ epochs: epochs, lr: lr })
        });
        const result = await res.json();
        
        if (result.success) {
            log(`FraudGCN model optimized successfully! Saved trained weights to trained_model.pt.`, "success");
            
            // Plot animated training loss curves
            plotLossCurve(result.losses);
            
            // Render KPI scores
            const m = result.metrics;
            dom.metricAcc.textContent = `${(m.accuracy * 100).toFixed(1)}%`;
            dom.metricPrec.textContent = `${(m.precision * 100).toFixed(1)}%`;
            dom.metricRec.textContent = `${(m.recall * 100).toFixed(1)}%`;
            dom.metricF1.textContent = `${(m.f1_score * 100).toFixed(1)}%`;
            
            log(`Model KPI Evaluation Metrics: Accuracy=${m.accuracy}, Precision=${m.precision}, Recall (Sensitivity)=${m.recall}, F1-Score=${m.f1_score}`, "success");
            
            dom.trainingProgress.textContent = "COMPLETED";
            dom.trainingProgress.classList.remove("training");
            
            checkSystemStatus();
        } else {
            log(`Training crash: ${result.error}`, "error");
            dom.trainingProgress.textContent = "ERROR";
            dom.trainingProgress.classList.remove("training");
        }
    } catch (e) {
        log(`API Communication error: ${e.message}`, "error");
        dom.trainingProgress.textContent = "ERROR";
        dom.trainingProgress.classList.remove("training");
    } finally {
        setButtonLoading(dom.btnTrainModel, false, "Train FraudGCN");
    }
});

// Auxiliary Loader buttons control
function setButtonLoading(btn, isLoading, text) {
    if (isLoading) {
        btn.setAttribute("disabled", "true");
        btn.classList.add("loading");
        btn.innerHTML = `<i class="fa-solid fa-spinner"></i> ${text}`;
    } else {
        btn.removeAttribute("disabled");
        btn.classList.remove("loading");
        btn.innerHTML = text;
    }
}

// 4. SUSPECTS LEDGER DIRECTORY
async function loadSuspectsDirectory() {
    try {
        const res = await fetch(`${API_BASE}/suspects`);
        const result = await res.json();
        
        if (result.success) {
            allSuspects = result.suspects;
            renderSuspectsTable();
            log(`Successfully synced Suspect Ledger: loaded ${allSuspects.length.toLocaleString()} node evaluations.`, "info");
        } else {
            log(`Failed loading risk directory: ${result.error}`, "error");
        }
    } catch (e) {
        log(`Failed fetching risk ledger: ${e.message}`, "error");
    }
}

function renderSuspectsTable() {
    dom.suspectsTableBody.innerHTML = "";
    
    const query = dom.suspectSearch.value.trim().toLowerCase();
    
    const filtered = allSuspects.filter(s => {
        // Search filter
        const matchSearch = s.name.toLowerCase().includes(query) || String(s.id).includes(query);
        if (!matchSearch) return false;
        
        // Category filters
        if (activeFilter === "critical") return s.score >= 0.75;
        if (activeFilter === "warning") return s.score >= 0.2 && s.score < 0.75;
        if (activeFilter === "legit") return s.score < 0.2;
        return true;
    });
    
    if (filtered.length === 0) {
        dom.suspectsTableBody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center text-muted">No matching suspect nodes found in the ledger.</td>
            </tr>
        `;
        return;
    }
    
    // Render top 100 for browser DOM performance
    const renderLimit = filtered.slice(0, 100);
    
    renderLimit.forEach(s => {
        const row = document.createElement("tr");
        if (activeSuspectId === s.id) {
            row.style.background = "rgba(56, 189, 248, 0.08)";
            row.style.borderColor = "var(--color-primary)";
        }
        
        // Threat Badge
        let badgeClass = "success";
        let badgeLabel = "Legit";
        if (s.score >= 0.75) {
            badgeClass = "danger";
            badgeLabel = "Critical";
        } else if (s.score >= 0.2) {
            badgeClass = "warning";
            badgeLabel = "Warning";
        }
        
        row.innerHTML = `
            <td><strong>${s.id}</strong></td>
            <td>${s.name}</td>
            <td><span class="badge ${badgeClass}">${s.role}</span></td>
            <td>$${s.avg_amount.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
            <td><strong style="color: ${getScoreColor(s.score)}">${(s.score * 100).toFixed(1)}%</strong></td>
            <td>
                <button class="btn-icon-circle" onclick="investigateNode(${s.id})" title="Investigate Ecosystem">
                    <i class="fa-solid fa-crosshairs"></i>
                </button>
            </td>
        `;
        dom.suspectsTableBody.appendChild(row);
    });
    
    if (filtered.length > 100) {
        const endRow = document.createElement("tr");
        endRow.innerHTML = `
            <td colspan="6" class="text-center text-muted" style="font-size: 0.72rem; padding: 8px;">
                Showing top 100 matches (Filtered total: ${filtered.length})
            </td>
        `;
        dom.suspectsTableBody.appendChild(endRow);
    }
}

// Search and filter listeners
dom.suspectSearch.addEventListener("input", renderSuspectsTable);

dom.filterBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        dom.filterBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        activeFilter = btn.getAttribute("data-filter");
        renderSuspectsTable();
    });
});

function getScoreColor(score) {
    if (score >= 0.75) return "#ef4444"; // Red
    if (score >= 0.2) return "#f59e0b"; // Amber
    return "#10b981"; // Green
}

// 5. GRAPH INVESTIGATION ENGINE
async function investigateNode(nodeId) {
    activeSuspectId = nodeId;
    const hops = dom.inputHops.value;
    
    log(`Opening forensic GNN ego-graph for suspect account ID ${nodeId} (${hops}-Hops depth)...`, "info");
    
    // Highlight table row
    renderSuspectsTable();
    
    try {
        const res = await fetch(`${API_BASE}/investigate/${nodeId}?hops=${hops}`);
        const result = await res.json();
        
        if (result.success) {
            // Find suspect profile object
            const suspect = allSuspects.find(s => s.id === nodeId);
            
            // Build suspect forensic details profile card
            renderSuspectProfileCard(suspect, result.nodes.length, result.edges.length);
            
            // Compile Vis.js datasets
            plotEgoGraphNetwork(result.nodes, result.edges);
            
            log(`Forensic sub-graph loaded! Plotted ${result.nodes.length} transacting users/transactions and ${result.edges.length} flow linkages.`, "success");
        } else {
            log(`Forensic sub-graph extraction crashed: ${result.error}`, "error");
        }
    } catch (e) {
        log(`API Communication error: ${e.message}`, "error");
    }
}

window.investigateNode = investigateNode; // Expose globally for onclick button actions

dom.inputHops.addEventListener("change", () => {
    if (activeSuspectId !== null) {
        investigateNode(activeSuspectId);
    }
});

function renderSuspectProfileCard(suspect, nodeCount, edgeCount) {
    if (!suspect) return;
    
    let dangerClass = "";
    if (suspect.score >= 0.75) dangerClass = "danger";
    else if (suspect.score >= 0.2) dangerClass = "warning";
    
    dom.investigateMetaPane.innerHTML = `
        <div class="suspect-profile">
            <div class="profile-field">
                <div class="profile-title">Investigating Actor</div>
                <div class="profile-value">${suspect.name} (ID: ${suspect.id})</div>
            </div>
            <div class="profile-field">
                <div class="profile-title">GNN Probability Threat</div>
                <div class="profile-value ${dangerClass}">${(suspect.score * 100).toFixed(2)}%</div>
            </div>
            <div class="profile-field">
                <div class="profile-title">Structural Pattern</div>
                <div class="profile-value">${suspect.role}</div>
            </div>
            <div class="profile-field">
                <div class="profile-title">Ego Neighborhood Dimensions</div>
                <div class="profile-value" style="font-size: 0.8rem; color: var(--text-secondary)">
                    ${nodeCount} nodes, ${edgeCount} tx paths
                </div>
            </div>
        </div>
    `;
}

// 6. SCIENTIFIC VISUAL CHARTS & GRAPHS
function plotLossCurve(losses) {
    const ctx = document.getElementById("lossChart").getContext("2d");
    
    const labels = losses.map(l => l.epoch);
    const data = losses.map(l => l.loss);
    
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'weighted Negative Log Likelihood (NLL) Loss',
                data: data,
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.05)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.2,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { family: 'Inter', size: 9 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { family: 'Inter', size: 9 } }
                }
            }
        }
    });
}

function plotEgoGraphNetwork(nodes, edges) {
    const container = document.getElementById("network-canvas");
    
    // Format Nodes for Vis.js
    const visNodes = nodes.map(n => {
        let nodeColor = "#3b82f6"; // default normal blue
        let nodeBorder = "#1d4ed8";
        let shape = "dot";
        let size = 15;
        
        // Format role colors
        if (n.role === "Fraud Boss") {
            nodeColor = "#ef4444"; // Red
            nodeBorder = "#b91c1c";
            size = 25;
        } else if (n.role === "Smurf") {
            nodeColor = "#f59e0b"; // Amber
            nodeBorder = "#b45309";
            size = 18;
        } else if (n.role === "Fraud Hub") {
            nodeColor = "#ec4899"; // Pink
            nodeBorder = "#be185d";
            size = 25;
        } else if (n.role === "Criminal Outflow") {
            nodeColor = "#a855f7"; // Purple
            nodeBorder = "#7e22ce";
            size = 20;
        } else if (n.role === "Collector Inflow") {
            nodeColor = "#eab308"; // Yellow
            nodeBorder = "#a16207";
            size = 20;
        } else if (n.role === "Transaction") {
            nodeColor = "#1e293b"; // Obsidian
            nodeBorder = "#64748b";
            shape = "diamond";
            size = 10;
        }
        
        // Highlight active target
        if (n.is_target) {
            nodeBorder = "#38bdf8";
            size = size + 6;
        }
        
        let labelText = n.label;
        if (n.role !== "Transaction") {
            labelText += `\nRisk: ${(n.score * 100).toFixed(0)}%`;
        }
        
        return {
            id: n.id,
            label: labelText,
            shape: shape,
            size: size,
            color: {
                background: nodeColor,
                border: nodeBorder,
                highlight: {
                    background: nodeColor,
                    border: "#38bdf8"
                }
            },
            font: {
                color: "#e2e8f0",
                size: 11,
                face: "Inter",
                strokeWidth: 2,
                strokeColor: "#0b0f19"
            },
            title: `
                <strong>Node ID:</strong> ${n.id}<br>
                <strong>Role:</strong> ${n.role}<br>
                <strong>Fraud Score:</strong> ${(n.score * 100).toFixed(2)}%<br>
                <strong>Avg Sent Amount:</strong> $${n.avg_amount.toLocaleString()}
            `
        };
    });
    
    // Format Edges
    const maxWeight = Math.max(...edges.map(e => e.amount), 1);
    
    const visEdges = edges.map(e => {
        // Edge thickness logic: min=1px, max=6px
        const thickness = 1.0 + 5.0 * (e.amount / maxWeight);
        
        return {
            from: e.from,
            to: e.to,
            width: thickness,
            arrows: {
                to: { enabled: true, scaleFactor: 0.6 }
            },
            color: {
                color: "rgba(100, 116, 139, 0.4)",
                highlight: "#38bdf8"
            },
            title: `Tx Value: $${e.amount.toLocaleString()}`
        };
    });
    
    const data = {
        nodes: new vis.DataSet(visNodes),
        edges: new vis.DataSet(visEdges)
    };
    
    const options = {
        physics: {
            stabilization: {
                enabled: true,
                iterations: 150
            },
            barnesHut: {
                gravitationalConstant: -2000,
                centralGravity: 0.3,
                springLength: 95,
                springConstant: 0.04
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 100
        }
    };
    
    if (networkEngine) {
        networkEngine.destroy();
    }
    
    networkEngine = new vis.Network(container, data, options);
    
    // Bind network events
    networkEngine.on("click", (params) => {
        if (params.nodes.length > 0) {
            const clickedId = params.nodes[0];
            
            // Check if node is a User (Transaction nodes aren't in allSuspects)
            const exists = allSuspects.some(s => s.id === clickedId);
            if (exists && clickedId !== activeSuspectId) {
                investigateNode(clickedId);
            }
        }
    });
}

// Initial Sync
checkSystemStatus();
log("AegisGNN System Ready. Connect your REST server to run forensic analytics.", "system");
