import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import re


base = os.path.dirname(os.path.abspath(__file__))
file_paths = [
    os.path.join(base, "MOSFET-data-n10.csv"),
    os.path.join(base, "MOSFET-data-n5.csv"),
    os.path.join(base, "MOSFET-data-n3.csv"),
    os.path.join(base, "MOSFET-data-n2.csv"),
    os.path.join(base, "MOSFET-data-n1.csv"),
]

file_specs = {
    "MOSFET-data-n1.csv":  {'W': 10e-6, 'L': 1e-6},
    "MOSFET-data-n2.csv":  {'W': 10e-6, 'L': 2e-6},
    "MOSFET-data-n3.csv":  {'W': 10e-6, 'L': 3e-6},
    "MOSFET-data-n5.csv":  {'W': 10e-6, 'L': 5e-6},
    "MOSFET-data-n10.csv": {'W': 10e-6, 'L': 10e-6},
}

VDS_LIN = 1.0  # Linear
VDS_SAT = 4.0  # Saturation
all_inputs = []  # W, L, Vgs, Vds
all_targets = [] # log10(Id)

for fpath in file_paths:
    fname = os.path.basename(fpath)
    if fname not in file_specs: continue
    
    W_val = file_specs[fname]['W']
    L_val = file_specs[fname]['L']
    
    try:
        df = pd.read_csv(fpath, header=[0, 1])
        new_cols = []
        last_reg = ""
        for c in df.columns:
            r, v = c
            if "Unnamed" not in str(r) and str(r) != "nan": last_reg = str(r).strip()
            new_cols.append((last_reg, str(v).strip()))
        df.columns = pd.MultiIndex.from_tuples(new_cols)

        for group in df.columns.levels[0]:
            group_upper = str(group).upper()
            
            if "LIN" in group_upper or "SAT" in group_upper:
                current_vds = VDS_LIN if "LIN" in group_upper else VDS_SAT
                
                if (group, 'Gate') in df.columns and (group, 'Id') in df.columns:
                    vgs = pd.to_numeric(df[(group, 'Gate')], errors='coerce').values
                    ids = pd.to_numeric(df[(group, 'Id')], errors='coerce').values
                    
                    mask = ~np.isnan(vgs) & ~np.isnan(ids) & (ids > 1e-14)
                    if np.sum(mask) > 0:
                        for v, i in zip(vgs[mask], ids[mask]):
                            all_inputs.append([W_val, L_val, v, current_vds])
                            all_targets.append(np.log10(i))

            elif "VD" in group_upper or "OUT" in group_upper:
                current_vgs = 3.0
                if (group, 'Gate') in df.columns:
                     pass
                 
                vd_col = 'Drain' if (group, 'Drain') in df.columns else 'Vd'
                if (group, vd_col) in df.columns and (group, 'Id') in df.columns:
                    vds_arr = pd.to_numeric(df[(group, vd_col)], errors='coerce').values
                    ids_arr = pd.to_numeric(df[(group, 'Id')], errors='coerce').values
                    
                    if "1" in group: current_vgs = 2.0
                    elif "2" in group: current_vgs = 3.0
                    elif "3" in group: current_vgs = 4.0
                    else: current_vgs = 3.0
                    
                    mask = ~np.isnan(vds_arr) & ~np.isnan(ids_arr) & (ids_arr > 1e-14)
                    for v, i in zip(vds_arr[mask], ids_arr[mask]):
                        all_inputs.append([W_val, L_val, current_vgs, v]) # Vgs는 고정, Vds는 변수
                        all_targets.append(np.log10(i))
        
    except Exception as e:
        print(f"  [Error] {fname}: {e}")

X_data = np.array(all_inputs, dtype=np.float32)
y_data = np.array(all_targets, dtype=np.float32).reshape(-1, 1)

mean_x = np.mean(X_data, axis=0)
std_x = np.std(X_data, axis=0)
std_x[std_x == 0] = 1.0

X_scaled = (X_data - mean_x) / std_x

indices = np.arange(len(X_data))
np.random.shuffle(indices)
split = int(len(X_data) * 0.8)

train_idx, test_idx = indices[:split], indices[split:]
X_train, y_train = X_scaled[train_idx], y_data[train_idx]
X_test, y_test = X_scaled[test_idx], y_data[test_idx]

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=512, shuffle=True)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test)

class MosfetNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

model = MosfetNN()
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.MSELoss()

loss_history = []
for epoch in range(1000):
    model.train()
    batch_loss = 0
    for bx, by in train_loader:
        pred = model(bx)
        loss = criterion(pred, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
    
    avg_loss = batch_loss / len(train_loader)
    loss_history.append(avg_loss)
    if epoch % 100 == 0: print(f"Epoch {epoch}: Loss {avg_loss:.5f}")

model.eval()

def predict_custom(W, L, Vgs, Vds):
    x = np.array([W, L, Vgs, Vds], dtype=np.float32)
    x_scaled = (x - mean_x) / std_x
    x_t = torch.tensor(x_scaled).view(1, -1)
    model.eval()
    with torch.no_grad():
        log_id = model(x_t).item()
    return 10**log_id

def Id(L_um, Vgs, W_um=10.0, Vds=4.0):
    W = W_um * 1e-6
    L = L_um * 1e-6
    return predict_custom(W, L, Vgs, Vds)

Id_val = Id(float(input("L = ")), float(input("Gate = ")))
print("Id =", Id_val*10e6, "uA")