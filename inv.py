import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
from scipy.optimize import brentq

# ==============================================================================
# 1. 공통 클래스 및 설정 정의
# ==============================================================================

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

class NeuralMOS_Wrapper:
    """
    학습된 모델과 정규화 분포(mean, std)를 함께 관리하는 클래스
    """
    def __init__(self, model, mean_x, std_x, device='cpu'):
        self.model = model
        self.mean_x = mean_x
        self.std_x = std_x
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict_Id(self, W, L, Vgs, Vds):
        # 입력 벡터 생성 (numpy)
        x = np.array([W, L, Vgs, Vds], dtype=np.float32)
        # 정규화
        x_scaled = (x - self.mean_x) / self.std_x
        # Tensor 변환
        x_t = torch.tensor(x_scaled, dtype=torch.float32).view(1, -1).to(self.device)
        
        with torch.no_grad():
            log_id = self.model(x_t).item()
        
        # log10(Id) -> Id (절대값 반환)
        return 10 ** log_id

def train_mosfet_model(file_paths, file_specs, device_type='n', epochs=1000):
    """
    device_type: 'n' for NMOS, 'p' for PMOS
    데이터를 로드하고 학습하여 Wrapper 객체를 반환합니다.
    """
    print(f"\n[{'NMOS' if device_type=='n' else 'PMOS'} Training Start]")
    
    # 전압 설정 (PMOS는 음수 전압 사용)
    pol = 1.0 if device_type == 'n' else -1.0
    
    VDS_LIN = 1.0 * pol
    VDS_SAT = 4.0 * pol
    
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
                
                # (1) LIN & SAT 데이터 (Id-Vg)
                if "LIN" in group_upper or "SAT" in group_upper:
                    current_vds = VDS_LIN if "LIN" in group_upper else VDS_SAT
                    
                    if (group, 'Gate') in df.columns and (group, 'Id') in df.columns:
                        vgs = pd.to_numeric(df[(group, 'Gate')], errors='coerce').values
                        ids = pd.to_numeric(df[(group, 'Id')], errors='coerce').values
                        ids = np.abs(ids) # 전류 절대값
                        
                        mask = ~np.isnan(vgs) & ~np.isnan(ids) & (ids > 1e-14)
                        if np.sum(mask) > 0:
                            for v, i in zip(vgs[mask], ids[mask]):
                                all_inputs.append([W_val, L_val, v, current_vds])
                                all_targets.append(np.log10(i))

                # (2) VD 데이터 (Id-Vd)
                elif "VD" in group_upper or "OUT" in group_upper:
                    # PMOS일 경우 Gate 전압도 음수여야 함
                    current_vgs = 3.0 * pol 
                    if "1" in group: current_vgs = 2.0 * pol
                    elif "2" in group: current_vgs = 3.0 * pol
                    elif "3" in group: current_vgs = 4.0 * pol
                    
                    vd_col = 'Drain' if (group, 'Drain') in df.columns else 'Vd'
                    if (group, vd_col) in df.columns and (group, 'Id') in df.columns:
                        vds_arr = pd.to_numeric(df[(group, vd_col)], errors='coerce').values
                        ids_arr = pd.to_numeric(df[(group, 'Id')], errors='coerce').values
                        ids_arr = np.abs(ids_arr) # 전류 절대값
                        
                        mask = ~np.isnan(vds_arr) & ~np.isnan(ids_arr) & (ids_arr > 1e-14)
                        for v, i in zip(vds_arr[mask], ids_arr[mask]):
                            all_inputs.append([W_val, L_val, current_vgs, v])
                            all_targets.append(np.log10(i))
        except Exception as e:
            print(f"  [Error] {fname}: {e}")

    # 데이터셋 구성
    X_data = np.array(all_inputs, dtype=np.float32)
    y_data = np.array(all_targets, dtype=np.float32).reshape(-1, 1)

    mean_x = np.mean(X_data, axis=0)
    std_x = np.std(X_data, axis=0)
    std_x[std_x == 0] = 1.0

    X_scaled = (X_data - mean_x) / std_x
    
    # 학습
    dataset = TensorDataset(torch.tensor(X_scaled), torch.tensor(y_data))
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    model = MosfetNN()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        batch_loss = 0
        for bx, by in loader:
            pred = model(bx)
            loss = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
        
        if epoch % 200 == 0:
            avg_loss = batch_loss / len(loader)
            print(f"  Epoch {epoch}: Loss {avg_loss:.5f}")

    print(f"  Done. Final Loss: {batch_loss / len(loader):.5f}")
    
    # Wrapper 객체로 포장해서 반환
    return NeuralMOS_Wrapper(model, mean_x, std_x)


def solve_inverter_voltage(vin, vdd, nmos_sim, pmos_sim, Wn, Ln, Wp, Lp):
    """
    주어진 Vin에 대해 I_nmos = I_pmos가 되는 Vout을 찾습니다.
    """
    def current_diff(vout):
        # NMOS: Vgs = Vin, Vds = Vout
        i_n = nmos_sim.predict_Id(Wn, Ln, vin, vout)
        
        # PMOS: Vgs = Vin - Vdd, Vds = Vout - Vdd
        vgs_p = vin - vdd
        vds_p = vout - vdd
        
        i_p = pmos_sim.predict_Id(Wp, Lp, vgs_p, vds_p)
        return i_n - i_p

    try:
        # 0 ~ VDD 사이에서 해 찾기
        vout_solution = brentq(current_diff, 0.0, vdd)
    except ValueError:
        # 해를 못 찾은 경우 (철저히 On/Off 영역)
        diff_0 = current_diff(0.001)
        diff_1 = current_diff(vdd - 0.001)
        vout_solution = 0.0 if abs(diff_0) < abs(diff_1) else vdd
            
    return vout_solution

# ==============================================================================
# 2. 메인 실행부
# ==============================================================================

if __name__ == "__main__":
    
    # --- 파일 경로 설정 ---
    base_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
    
    nmos_files = [base_path + f"MOSFET-data-n{x}.csv" for x in [10, 5, 3, 2, 1]]
    pmos_files = [base_path + f"MOSFET-data-p{x}.csv" for x in [10, 5, 3, 2, 1]]
    
    common_specs = {
        "MOSFET-data-n1.csv":  {'W': 10e-6, 'L': 1e-6}, "MOSFET-data-p1.csv":  {'W': 10e-6, 'L': 1e-6},
        "MOSFET-data-n2.csv":  {'W': 10e-6, 'L': 2e-6}, "MOSFET-data-p2.csv":  {'W': 10e-6, 'L': 2e-6},
        "MOSFET-data-n3.csv":  {'W': 10e-6, 'L': 3e-6}, "MOSFET-data-p3.csv":  {'W': 10e-6, 'L': 3e-6},
        "MOSFET-data-n5.csv":  {'W': 10e-6, 'L': 5e-6}, "MOSFET-data-p5.csv":  {'W': 10e-6, 'L': 5e-6},
        "MOSFET-data-n10.csv": {'W': 10e-6, 'L': 10e-6},"MOSFET-data-p10.csv": {'W': 10e-6, 'L': 10e-6},
    }

    # --- 1. NMOS 학습 및 저장 ---
    nmos_unit = train_mosfet_model(nmos_files, common_specs, device_type='n', epochs=1000)

    # --- 2. PMOS 학습 및 저장 ---
    pmos_unit = train_mosfet_model(pmos_files, common_specs, device_type='p', epochs=1000)

    # --- 3. 인버터 시뮬레이션 ---
    print("\n[Inverter Simulation Start]")
    VDD = 4.0  # Supply Voltage
    W_n = 10e-6; L_n = 1e-6
    W_p = 20e-6; L_p = 1e-6  # PMOS Width 2배

    vin_sweep = np.linspace(0, VDD, 100)
    vout_list = []
    current_list = []

    for vin in vin_sweep:
        # VTC 계산
        vout = solve_inverter_voltage(vin, VDD, nmos_unit, pmos_unit, W_n, L_n, W_p, L_p)
        vout_list.append(vout)
        
        # 관통 전류 (Short Circuit Current) 계산
        ids = nmos_unit.predict_Id(W_n, L_n, vin, vout)
        current_list.append(ids)

    # --- 4. 결과 시각화 ---
    plt.figure(figsize=(12, 5))

    # VTC Curve
    plt.subplot(1, 2, 1)
    plt.plot(vin_sweep, vout_list, 'b-', linewidth=2, label='NN VTC')
    plt.plot([0, VDD], [0, VDD], 'k--', alpha=0.3) # y=x line
    plt.plot([VDD/2, VDD/2], [0, VDD], 'g--', alpha=0.3) # Switching Threshold check
    plt.title(f"CMOS Inverter VTC (VDD={VDD}V)")
    plt.xlabel("Input Voltage ($V_{in}$) [V]")
    plt.ylabel("Output Voltage ($V_{out}$) [V]")
    plt.grid(True)
    plt.legend()

    # Current Curve
    plt.subplot(1, 2, 2)
    plt.plot(vin_sweep, np.array(current_list)*1e6, 'r-', linewidth=2)
    plt.title("Short Circuit Current")
    plt.xlabel("Input Voltage ($V_{in}$) [V]")
    plt.ylabel("Current ($I_d$) [$\mu A$]")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    print("Simulation Finished.")