import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_bank_data(num_users=1000, num_normal_tx=5000):
    print("Initialize banking simulator...")
    
    # 1. Define Nodes: Users, Master Criminals, and Collectors
    print("Designating Special Roles (Criminals and Collectors)...")
    num_criminals = int(num_users * 0.05)
    num_collectors = int(num_users * 0.05)
    num_normal_users = num_users - num_criminals - num_collectors
    
    criminals = [f"U_CRIM_{i}" for i in range(num_criminals)]
    collectors = [f"U_COLL_{i}" for i in range(num_collectors)]
    normal_users = [f"U_NORM_{i}" for i in range(num_normal_users)]
    
    # We create two distinct isolated clusters of normal users, A and B
    cluster_a = normal_users[:len(normal_users)//2]
    cluster_b = normal_users[len(normal_users)//2:]
    
    all_users = criminals + collectors + normal_users
    transactions = []
    start_date = datetime(2026, 1, 1)
    
    # Helper to generate random timestamps
    def get_random_time():
        days_offset = random.randint(0, 30)
        hours = random.randint(0, 23)
        mins = random.randint(0, 59)
        return start_date + timedelta(days=days_offset, hours=hours, minutes=mins)

    tx_counter = 0

    # 2. Normal Transactions (Strictly staying within their own clusters mostly, to build shape!)
    print(f"Generating {num_normal_tx} normal transactions inside isolated clusters...")
    for _ in range(num_normal_tx):
        # 50/50 chance for a transaction to occur in Cluster A or Cluster B
        cluster = cluster_a if random.random() > 0.5 else cluster_b
        
        sender = random.choice(cluster)
        receiver = random.choice(cluster)
        while sender == receiver:
            receiver = random.choice(cluster)
            
        amount = round(random.uniform(5.0, 800.0), 2)
        
        transactions.append({
            'tx_id': f"TX_{tx_counter}",
            'sender': sender,
            'receiver': receiver,
            'amount': amount,
            'timestamp': get_random_time().strftime('%Y-%m-%d %H:%M:%S'),
            'is_fraud': 0,
            'type': 'normal'
        })
        tx_counter += 1
        
    # 3. MASTER CRIMINALS (High Out-Degree): Send money to MANY normal users
    print("Programming Master Criminals (High Out-Degree)...")
    for criminal in criminals:
        # Each criminal sends to 10-30 different people
        targets = random.sample(normal_users, random.randint(10, 30))
        for target in targets:
            transactions.append({
                'tx_id': f"TX_{tx_counter}",
                'sender': criminal,
                'receiver': target,
                'amount': round(random.uniform(500.0, 2000.0), 2),
                'timestamp': get_random_time().strftime('%Y-%m-%d %H:%M:%S'),
                'is_fraud': 1,  # Label as fraud!
                'type': 'criminal_outflow'
            })
            tx_counter += 1

    # 4. COLLECTORS (High In-Degree): Receive money from MANY normal users
    print("Programming Collectors (High In-Degree)...")
    for collector in collectors:
        # Each collector receives from 10-30 different people
        sources = random.sample(normal_users, random.randint(10, 30))
        for source in sources:
            transactions.append({
                'tx_id': f"TX_{tx_counter}",
                'sender': source,
                'receiver': collector,
                'amount': round(random.uniform(5000.0, 10000.0), 2), # Larger sums
                'timestamp': get_random_time().strftime('%Y-%m-%d %H:%M:%S'),
                'is_fraud': 1,  # Label as fraud!
                'type': 'collector_inflow'
            })
            tx_counter += 1

    # 5. THE BRIDGE EFFECT 
    print("Building 'Bridges' connecting Isolated Clusters A and B...")
    num_bridges = 20
    for _ in range(num_bridges):
        # Transaction forcibly crossing Cluster A -> Cluster B
        sender = random.choice(cluster_a)
        receiver = random.choice(cluster_b)
        transactions.append({
            'tx_id': f"TX_{tx_counter}",
            'sender': sender,
            'receiver': receiver,
            'amount': round(random.uniform(10.0, 100.0), 2),
            'timestamp': get_random_time().strftime('%Y-%m-%d %H:%M:%S'),
            'is_fraud': 0, # Normal bridge
            'type': 'bridge'
        })
        tx_counter += 1

    # 6. Fraud Injection: Smurfing (Money Laundering Ring)
    print("Injecting Smurfing Ring (1 boss -> 100 small transactions -> 1 hub)...")
    fraud_source = "U_FRAUD_BOSS"
    num_smurfs = 100
    smurfs = [f"U_SMURF_{i}" for i in range(num_smurfs)]
    fraud_hub = "U_FRAUD_HUB"
    
    # Boss sends $100 to 100 smurfs
    smurf_amount = 100.0
    smurf_time = start_date + timedelta(days=15, hours=12)
    
    for i, smurf in enumerate(smurfs):
        # Boss -> Smurf
        transactions.append({
            'tx_id': f"TX_{tx_counter}",
            'sender': fraud_source,
            'receiver': smurf,
            'amount': smurf_amount,
            'timestamp': (smurf_time + timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S'),
            'is_fraud': 1,
            'type': 'smurfing_layer1'
        })
        tx_counter += 1
        
        # Layer 2: Smurfs -> Hub
        transactions.append({
            'tx_id': f"TX_{tx_counter}",
            'sender': smurf,
            'receiver': fraud_hub,
            'amount': smurf_amount * random.uniform(0.9, 0.99),
            'timestamp': (smurf_time + timedelta(hours=2, minutes=random.randint(0, 60))).strftime('%Y-%m-%d %H:%M:%S'),
            'is_fraud': 1,
            'type': 'smurfing_layer2'
        })
        tx_counter += 1
        
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(transactions)
    
    # Sort by timestamp to simulate real transaction log!
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    df.to_csv("data/raw_transactions.csv", index=False)
    print(f"Generated {len(df)} total transactions into data/raw_transactions.csv")
    return df

if __name__ == '__main__':
    generate_bank_data()
