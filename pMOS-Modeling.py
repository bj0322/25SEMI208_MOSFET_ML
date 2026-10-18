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

# =========================================================
# 1. 파일 설정 (PMOS)
# =========================================================
file_paths = [
    str(DATA_DIR / "MOSFET-data-p10.csv"),
    str(DATA_DIR / "MOSFET-data-p5.csv"),
    str(DATA_DIR / "MOSFET-data-p3.csv"),
    str(DATA_DIR / "MOSFET-data-p2.csv"),
    str(DATA_DIR / "MOSFET-data-p1.csv")
]

file_specs = {
    "MOSFET-data-p1.csv":  {'W': 10e-6, 'L': 1e-6},
    "MOSFET-data-p2.csv":  {'W': 10e-6, 'L': 2e-6},
    "MOSFET-data-p3.csv":  {'W': 10e-6, 'L': 3e-6},
    "MOSFET-data-p5.csv":  {'W': 10e-6, 'L': 5e-6},
    "MOSFET-data-p10.csv": {'W': 10e-6, 'L': 10e-6},
}

VDS_LIN = -1.0 
VDS_SAT = -4.0 

all_inputs = [] 
all_targets = [] 

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
            
            # (1) LIN & SAT 데이터 처리 (Id-Vg)
            if "LIN" in group_upper or "SAT" in group_upper:
                current_vds = VDS_LIN if "LIN" in group_upper else VDS_SAT
                
                if (group, 'Gate') in df.columns and (group, 'Id') in df.columns:
                    vgs = pd.to_numeric(df[(group, 'Gate')], errors='coerce').values
                    ids = pd.to_numeric(df[(group, 'Id')], errors='coerce').values
                    
                    # [핵심 수정 1] 전류 절대값 처리 (PMOS 전류가 음수일 경우 대비)
                    ids = np.abs(ids)
                    
                    # 유효 데이터 필터링
                    mask = ~np.isnan(vgs) & ~np.isnan(ids) & (ids > 1e-14)
                    
                    if np.sum(mask) > 0:
                        for v, i in zip(vgs[mask], ids[mask]):
                            all_inputs.append([W_val, L_val, v, current_vds])
                            all_targets.append(np.log10(i))

            # (2) VD 데이터 처리 (Id-Vd)
            elif "VD" in group_upper or "OUT" in group_upper:
                # [핵심 수정 2] PMOS는 Gate 전압도 음수여야 켜집니다.
                # 파일의 측정 조건에 맞춰 음수로 변경 (예: -2V, -3V, -4V)
                current_vgs = -3.0 
                if "1" in group: current_vgs = -2.0
                elif "2" in group: current_vgs = -3.0
                elif "3" in group: current_vgs = -4.0
                else: current_vgs = -3.0
                
                vd_col = 'Drain' if (group, 'Drain') in df.columns else 'Vd'
                if (group, vd_col) in df.columns and (group, 'Id') in df.columns:
                    vds_arr = pd.to_numeric(df[(group, vd_col)], errors='coerce').values
                    ids_arr = pd.to_numeric(df[(group, 'Id')], errors='coerce').values
                    
                    # 전류 절대값
                    ids_arr = np.abs(ids_arr)
                    
                    mask = ~np.isnan(vds_arr) & ~np.isnan(ids_arr) & (ids_arr > 1e-14)
                    for v, i in zip(vds_arr[mask], ids_arr[mask]):
                        all_inputs.append([W_val, L_val, current_vgs, v])
                        all_targets.append(np.log10(i))
        
    except Exception as e:
        print(f"  [Error] {fname}: {e}")

# 배열 변환
X_data = np.array(all_inputs, dtype=np.float32)
y_data = np.array(all_targets, dtype=np.float32).reshape(-1, 1)

# =========================================================
# 3. 데이터 전처리
# =========================================================
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
with torch.no_grad():
    y_pred_test = model(X_test_t).numpy()

X_test_original = (X_test * std_x) + mean_x

plt.figure(figsize=(14, 6))

# (1) Scatter Plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_test, s=2, alpha=0.3, color='blue', label='Test Data')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', linewidth=1.5, label='Ideal')
plt.title("Prediction vs Measured (PMOS)")
plt.xlabel("Measured log(|Id|)"); plt.ylabel("Predicted log(|Id|)")
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
sample_W = X_test_original[0, 0]
sample_L = X_test_original[0, 1]
sample_Vds = X_test_original[0, 3]

mask = (np.abs(X_test_original[:, 0] - sample_W) < 1e-9) & \
       (np.abs(X_test_original[:, 1] - sample_L) < 1e-9) & \
       (np.abs(X_test_original[:, 3] - sample_Vds) < 0.1)

if np.sum(mask) > 10:
    vg_sample = X_test_original[mask, 2]
    id_meas_sample = y_test[mask]
    
    plt.scatter(vg_sample, id_meas_sample, color='green', s=15, label='Measured')
    
    vg_sweep = np.linspace(min(vg_sample), max(vg_sample), 100)
    input_sweep = np.zeros((100, 4))
    input_sweep[:, 0] = sample_W
    input_sweep[:, 1] = sample_L
    input_sweep[:, 2] = vg_sweep
    input_sweep[:, 3] = sample_Vds
    
    input_sweep_scaled = (input_sweep - mean_x) / std_x
    with torch.no_grad():
        pred_sweep = model(torch.tensor(input_sweep_scaled, dtype=torch.float32)).numpy()
        
    plt.plot(vg_sweep, pred_sweep, 'r-', linewidth=2, label='NN Prediction')
    plt.title(f"I-V (L={sample_L*1e6:.1f}um, Vds={sample_Vds:.1f}V)")
else:
    plt.text(0.5, 0.5, "No sufficient samples")

plt.xlabel("Vgs (V)"); plt.ylabel("log(|Id|)")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

def predict_custom(W, L, Vgs, Vds):
    x = np.array([W, L, Vgs, Vds], dtype=np.float32)
    x_scaled = (x - mean_x) / std_x
    x_t = torch.tensor(x_scaled).view(1, -1)
    
    model.eval()
    with torch.no_grad():
        log_id = model(x_t).item()
        
    return 10**log_id

print("\n[PMOS 예측 테스트] W=10u, L=2u, Vgs=-3V, Vds=-4V")
print(f"|Id| = {predict_custom(10e-6, 1e-6, -3.0, -4.0):.3e} A")

# ============================================================
# PMOS level=3 스타일 파라미터 피팅 (VTO, KP, LAMBDA, THETA, ETA)
# - PMOS는 VSG=-VGS, VSD=-VDS로 바꿔 NMOS 형태로 magnitude(|Id|) 피팅
# ============================================================

# 1) log(|Id|) -> |Id| 로 복원
Id_mag_data = 10 ** y_data.flatten()  # [N] (이미 y_data는 log10(|Id|)로 만들었음)

W_all   = X_data[:, 0]
L_all   = X_data[:, 1]
Vgs_all = X_data[:, 2]   # PMOS 측정값: 보통 음수
Vds_all = X_data[:, 3]   # PMOS 측정값: 보통 음수

# (선택) 서브스레시홀드/노이즈 구간 제외 기준
mask = Id_mag_data > 1e-9

# PMOS on-state 데이터만 쓰고 싶으면(권장): Vgs<0, Vds<0 조건 추가
mask = mask & (Vgs_all < 0.0) & (Vds_all < 0.0)

W_t   = torch.tensor(W_all[mask],   dtype=torch.float32)
L_t   = torch.tensor(L_all[mask],   dtype=torch.float32)
Vgs_t = torch.tensor(Vgs_all[mask], dtype=torch.float32)
Vds_t = torch.tensor(Vds_all[mask], dtype=torch.float32)
Id_t  = torch.tensor(Id_mag_data[mask], dtype=torch.float32)


# 2) NMOS용 “크기” 모델 (VSG,VSD로 변환해서 사용)
def mos3_Id_mag_pmos(W, L, Vgs, Vds, VTO_abs, KP, LAMBDA, THETA, ETA):
    """
    VTO_abs : |VTP| (양수로 피팅)
    KP      : 양수
    입력(Vgs,Vds)은 PMOS 측정 부호(보통 음수)
    출력은 |Id|
    """
    # PMOS -> magnitude domain
    Vsg = torch.clamp(-Vgs, min=0.0)   # VSG = -VGS
    Vsd = torch.clamp(-Vds, min=0.0)   # VSD = -VDS

    # DIBL: |Vth| 감소를 magnitude에서 동일하게 반영
    Vth_eff = VTO_abs - ETA * Vsd
    Vgt = Vsg - Vth_eff
    Vgt = torch.clamp(Vgt, min=0.0)

    beta = KP * (W / L)

    lin_region = Vsd < Vgt
    sat_region = ~lin_region

    Id_lin = beta * (Vgt * Vsd - 0.5 * Vsd**2)
    Id_sat = 0.5 * beta * Vgt**2

    Id = torch.where(lin_region, Id_lin, Id_sat)

    # CLM
    Id = Id * (1.0 + LAMBDA * Vsd)

    # mobility degradation
    Id = Id / (1.0 + THETA * Vgt)

    return torch.clamp(Id, min=1e-15)


# 3) 피팅 파라미터(초기값은 “|VTP| 양수”로 두는 게 포인트)
VTO_abs = nn.Parameter(torch.tensor(1.2))      # |VTP|, 양수
KP      = nn.Parameter(torch.tensor(3e-5))     # A/V^2 (PMOS는 보통 NMOS보다 작게 시작)
LAMBDA  = nn.Parameter(torch.tensor(0.02))     # 1/V
THETA   = nn.Parameter(torch.tensor(0.3))      # 1/V
ETA     = nn.Parameter(torch.tensor(0.05))     # 1/V 정도로 시작

params = [VTO_abs, KP, LAMBDA, THETA, ETA]
optimizer_fit = torch.optim.Adam(params, lr=1e-2)

# 4) log10(|Id|) 기준 MSE 최소화
for epoch in range(2000):
    optimizer_fit.zero_grad()

    Id_pred = mos3_Id_mag_pmos(W_t, L_t, Vgs_t, Vds_t, VTO_abs, KP, LAMBDA, THETA, ETA)
    logId_pred = torch.log10(Id_pred)
    logId_true = torch.log10(Id_t)

    loss = torch.mean((logId_pred - logId_true) ** 2)
    loss.backward()
    optimizer_fit.step()

    # 물리적으로 말이 안 되는 방향으로 가는 걸 방지(가끔 필요)
    with torch.no_grad():
        VTO_abs.clamp_(0.1, 5.0)
        KP.clamp_(1e-8, 1e-2)
        LAMBDA.clamp_(0.0, 1.0)
        THETA.clamp_(0.0, 5.0)
        ETA.clamp_(0.0, 1.0)

    if epoch % 200 == 0:
        print(
            f"[PMOS FIT] Epoch {epoch:4d} "
            f"loss={loss.item():.4e}, "
            f"|VTP|={VTO_abs.item():.3f}, KP={KP.item():.3e}, "
            f"LAMBDA={LAMBDA.item():.3e}, THETA={THETA.item():.3f}, ETA={ETA.item():.3f}"
        )

# 5) 최종 .model 출력 (PMOS는 VTO를 음수로!)
print("\n===== Fitted PMOS SPICE model =====")
print(".model p_fit PMOS level=3")
print(f"+ VTO    = {-VTO_abs.item():.4g}")   # <- 여기만 음수!
print(f"+ KP     = {KP.item():.4g}")
print(f"+ GAMMA  = 0")
print(f"+ PHI    = 0.7")
print(f"+ LAMBDA = {LAMBDA.item():.4g}")
print(f"+ THETA  = {THETA.item():.4g}")
print(f"+ ETA    = {ETA.item():.4g}")
