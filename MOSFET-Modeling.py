import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path

# Resolve measurement data relative to this script for portable execution.
DATA_DIR = Path(__file__).resolve().parent

file_paths = [
    str(DATA_DIR / "MOSFET-data-n10.csv"),
    str(DATA_DIR / "MOSFET-data-n5.csv"),
    str(DATA_DIR / "MOSFET-data-n3.csv"),
    str(DATA_DIR / "MOSFET-data-n2.csv"),
    str(DATA_DIR / "MOSFET-data-n1.csv")
]

file_specs = {
    "MOSFET-data-n1.csv":  {'W': 10e-6, 'L': 1e-6},
    "MOSFET-data-n2.csv":  {'W': 10e-6, 'L': 2e-6},
    "MOSFET-data-n3.csv":  {'W': 10e-6, 'L': 3e-6},
    "MOSFET-data-n5.csv":  {'W': 10e-6, 'L': 5e-6},
    "MOSFET-data-n10.csv": {'W': 10e-6, 'L': 10e-6},
}

VDS_LIN = 1.0  # Linear 측정 Vds
VDS_SAT = 4.0  # Saturation 측정 Vds
all_inputs = []  # [W, L, Vgs, Vds]
all_targets = [] # [log10(Id)]

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

X_test_original = (X_test * std_x) + mean_x
L_list_um = [1, 2, 3, 5, 10]

W_fixed = 10e-6
Vds_target = 4.0

for L_um in L_list_um:
    L_val = L_um * 1e-6
    
    mask = (
        (np.abs(X_test_original[:, 0] - W_fixed) < 1e-9) &
        (np.abs(X_test_original[:, 1] - L_val)   < 1e-9) &
        (np.abs(X_test_original[:, 3] - Vds_target) < 0.1)
    )

    if np.sum(mask) < 10:
        continue

    vg_sample = X_test_original[mask, 2]
    id_meas_sample = y_test[mask].flatten()

    vg_min = float(np.min(vg_sample))
    vg_max = float(np.max(vg_sample))
    vg_sweep = np.linspace(vg_min, vg_max, 200)

    input_sweep = np.zeros((len(vg_sweep), 4), dtype=np.float32)
    input_sweep[:, 0] = W_fixed
    input_sweep[:, 1] = L_val
    input_sweep[:, 2] = vg_sweep
    input_sweep[:, 3] = Vds_target

    input_sweep_scaled = (input_sweep - mean_x) / std_x
    with torch.no_grad():
        pred_sweep = model(
            torch.tensor(input_sweep_scaled, dtype=torch.float32)
        ).numpy().flatten()

    plt.figure(figsize=(7, 5))

    plt.scatter(vg_sample, id_meas_sample,
                color='green', s=15, alpha=0.6, label='Measured (Test)')

    plt.plot(vg_sweep, pred_sweep,
             'r-', linewidth=2, label='NN Prediction')

    plt.title(f"I-V (L = {L_um:.1f} μm, W = {W_fixed*1e6:.1f} μm, Vds = {Vds_target:.1f} V)")
    plt.xlabel("Vgs (V)")
    plt.ylabel("log10(Id) (A)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"IV_L{L_um:.0f}um.png", dpi=300)
    plt.show()

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

Id_val = Id(2.0, 2.0)
print("Id =", Id_val, "A")

# ============================================================
# SPICE level=3 스타일 NMOS 파라미터 피팅 (VTO, KP, LAMBDA, THETA, ETA)
# ============================================================

# 1) log(Id) -> Id 로 되돌리고, 강한 inversion 영역만 사용해서 피팅
Id_data = 10 ** y_data.flatten()  # [N]  (y_data는 log10(Id))

W_all   = X_data[:, 0]
L_all   = X_data[:, 1]
Vgs_all = X_data[:, 2]
Vds_all = X_data[:, 3]

# 너무 작은 전류(서브스레시홀드)는 일단 제외 (원하는 기준으로 바꿔도 됨)
mask = Id_data > 1e-9

W_t   = torch.tensor(W_all[mask],   dtype=torch.float32)
L_t   = torch.tensor(L_all[mask],   dtype=torch.float32)
Vgs_t = torch.tensor(Vgs_all[mask], dtype=torch.float32)
Vds_t = torch.tensor(Vds_all[mask], dtype=torch.float32)
Id_t  = torch.tensor(Id_data[mask], dtype=torch.float32)


# 2) 단순화된 level=3 NMOS Id 모델 + DIBL(ETA) 포함
def mos3_Id(W, L, Vgs, Vds, VTO, KP, LAMBDA, THETA, ETA):
    Vds_clipped = torch.clamp(Vds, min=0.0)

    # DIBL: Vth_eff = VTO - ETA * Vds
    Vth_eff = VTO - ETA * Vds_clipped

    # gate overdrive
    Vgt = Vgs - Vth_eff
    Vgt_clipped = torch.clamp(Vgt, min=0.0)

    beta = KP * (W / L)  # KP * W/L

    # 영역 구분
    lin_region = Vds_clipped < Vgt_clipped
    sat_region = ~lin_region

    Id_lin = beta * (Vgt_clipped * Vds_clipped - 0.5 * Vds_clipped ** 2)
    Id_sat = 0.5 * beta * Vgt_clipped ** 2

    Id = torch.where(lin_region, Id_lin, Id_sat)

    # channel-length modulation
    Id = Id * (1.0 + LAMBDA * Vds_clipped)

    # mobility degradation
    Id = Id / (1.0 + THETA * Vgt_clipped)

    # 수치 안정화
    Id = torch.clamp(Id, min=1e-15)
    return Id


# 3) 피팅할 파라미터 정의 (초기값은 예시 SPICE 값 근처로)
VTO     = nn.Parameter(torch.tensor(1.2))      # V
KP      = nn.Parameter(torch.tensor(7e-5))     # A/V^2
LAMBDA  = nn.Parameter(torch.tensor(0.02))     # 1/V
THETA   = nn.Parameter(torch.tensor(0.386))    # 1/V
ETA     = nn.Parameter(torch.tensor(0.05))     # 1 (DIBL 계수, 대략적인 초기값)

params = [VTO, KP, LAMBDA, THETA, ETA]
optimizer_fit = torch.optim.Adam(params, lr=1e-2)

# 4) 최적화 루프 (log10(Id) 기준으로 MSE 최소화)
for epoch in range(2000):
    optimizer_fit.zero_grad()

    Id_pred = mos3_Id(W_t, L_t, Vgs_t, Vds_t, VTO, KP, LAMBDA, THETA, ETA)
    logId_pred = torch.log10(Id_pred)
    logId_true = torch.log10(Id_t)

    loss = torch.mean((logId_pred - logId_true) ** 2)
    loss.backward()
    optimizer_fit.step()

    if epoch % 200 == 0:
        print(
            f"[FIT] Epoch {epoch:4d}  "
            f"loss={loss.item():.4e},  "
            f"VTO={VTO.item():.3f},  KP={KP.item():.3e},  "
            f"LAMBDA={LAMBDA.item():.3e},  THETA={THETA.item():.3f},  "
            f"ETA={ETA.item():.3f}"
        )

# 5) 최종 .model 라인 출력
print("\n===== Fitted NMOS SPICE model =====")
print(".model n_fit NMOS level=3")
print(f"+ VTO    = {VTO.item():.4g}")
print(f"+ KP     = {KP.item():.4g}")
print(f"+ GAMMA  = 0")
print(f"+ PHI    = 0.6")
print(f"+ LAMBDA = {LAMBDA.item():.4g}")
print(f"+ THETA  = {THETA.item():.4g}")
print(f"+ ETA    = {ETA.item():.4g}")

# -------------------------------------------------------------------------
# [수정됨] Vds까지 변수로 포함하여 목표 전력 최적화 (Total Design Optimization)
# -------------------------------------------------------------------------
def optimize_mosfet_free_vds(target_power, fixed_W):

    opt_L = torch.tensor([5e-6], dtype=torch.float32, requires_grad=True)
    opt_Vgs = torch.tensor([3.0], dtype=torch.float32, requires_grad=True)
    opt_Vds = torch.tensor([3.0], dtype=torch.float32, requires_grad=True)
    optimizer_input = optim.Adam([opt_L, opt_Vgs, opt_Vds], lr=0.05)

    model.eval()
    
    # 정규화용 텐서
    mean_x_t = torch.tensor(mean_x)
    std_x_t = torch.tensor(std_x)
    fixed_W_t = torch.tensor(fixed_W)

    for step in range(2000): # 변수가 많아져서 step을 조금 늘림
        optimizer_input.zero_grad()

        # 3. 입력 벡터 구성 [W, L, Vgs, Vds]
        # 모든 변수가 텐서 그래프에 연결되어야 함
        current_input = torch.stack([
            fixed_W_t, 
            opt_L[0], 
            opt_Vgs[0], 
            opt_Vds[0]  # Vds가 변수로 들어감
        ])

        # 4. 정규화 및 예측
        scaled_input = (current_input - mean_x_t) / std_x_t
        pred_log_id = model(scaled_input.unsqueeze(0))
        
        # 5. 전력 계산 (P = Vds * Id)
        # 여기서 opt_Vds[0]가 사용되므로 Vds에 대해서도 미분이 발생함
        pred_id = torch.pow(10, pred_log_id)
        pred_power = pred_id * opt_Vds[0] 

        # 6. Loss 계산
        target_p_tensor = torch.tensor([target_power])
        loss = nn.MSELoss()(pred_power, target_p_tensor)

        # 7. 업데이트
        loss.backward()
        optimizer_input.step()

        # 8. 물리적 제약 조건 (Clamping)
        with torch.no_grad():
            opt_L.clamp_(1e-6, 10e-6)   # L: 1~10um
            opt_Vgs.clamp_(0.5, 5.0)    # Vgs: 0.5~5V (Vth 고려 최소값 설정)
            opt_Vds.clamp_(0.1, 5.0)    # Vds: 0.1~5V (0V가 되면 전력 계산 불가/무의미)

        if loss.item() < 1e-13:
            break

    return opt_L.item(), opt_Vgs.item(), opt_Vds.item(), pred_id.item()

# -------------------------------------------------------------------------
# [메인 실행부]
# -------------------------------------------------------------------------

print("\n" + "="*40)
print(" 1. Predict Id (Forward)")
print(" 2. Find Optimal Design (Auto L, Vgs, Vds)")
print("="*40)

mode = input("Select Mode (1 or 2): ")

if mode == "1":
    # (기존과 동일)
    try:
        in_L = float(input("L (um) = "))
        in_Vgs = float(input("Vgs (V) = "))
        in_W = float(input("W (um, default 10) = ") or 10.0)
        in_Vds = float(input("Vds (V, default 4.0) = ") or 4.0)

        result_Id = predict_custom(in_W*1e-6, in_L*1e-6, in_Vgs, in_Vds)
        print(f"\nResult:")
        print(f"  Id  = {result_Id*1e6:.4f} uA")
        print(f"  Pwr = {result_Id * in_Vds * 1e3:.4f} mW")
    except Exception as e:
        print("Invalid input:", e)

elif mode == "2":
    try:
        # Vds 입력 절차 삭제됨
        target_P_mW = float(input("Target Power (mW) = "))
        fixed_W_um = float(input("Fixed W (um) = ") or 10.0)
        
        target_P = target_P_mW * 1e-3
        fixed_W = fixed_W_um * 1e-6
        
        # 최적화 수행 (Vds 없음)
        best_L, best_Vgs, best_Vds, best_Id = optimize_mosfet_free_vds(target_P, fixed_W)
        print()
        print("소자 구조 파라미터 최적화")
        print(f"Target Power = {target_P_mW} mW")
        print(f"Static Power = {best_Id * best_Vds * 1e3:.4f} mW")
        print(f"L = {best_L*1e6:.4f} um")
        print(f"Vgs = {best_Vgs:.4f} V")
        print(f"Vds = {best_Vds:.4f} V")
        print(f"Id = {best_Id*1e6:.4f} uA")
        print("="*40)
        
    except ValueError:
        print("Invalid number format.")

print("\nProcess finished. Press Enter to exit.")
input()