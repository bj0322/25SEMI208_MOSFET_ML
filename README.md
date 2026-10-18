# 25SEMI208_MOSFET_ML

Measurement-driven MOSFET modeling and CMOS inverter analysis for a 2025 SEMI208 course project. This repository brings together the Python experiments, measured I–V data, spreadsheet exports, and supporting laboratory material from the original `2025_집공개II` project folder.

## Overview

- Train separate NMOS and PMOS neural networks to predict drain-current magnitude from device geometry and terminal voltages.
- Fit a simplified, Level-3-style current equation and print SPICE-format parameter estimates.
- Explore NMOS operating points for a target static power.
- Combine neural device models to estimate a CMOS inverter voltage-transfer characteristic and current curve.

These are educational experiments, not a validated compact-model implementation or a replacement for a circuit simulator.

## Repository contents

| Path | Purpose |
|---|---|
| `MOSFET-Modeling.py` | NMOS training, I–V plots, simplified parameter fitting, and interactive forward/inverse design experiments |
| `pMOS-Modeling.py` | PMOS current-magnitude training, plots, and parameter fitting |
| `inv.py` | Train both polarities and solve inverter current balance with SciPy's Brent root finder |
| `MOSFET-data-n*.csv`, `MOSFET-data-p*.csv` | Ten consolidated measurement tables consumed by the models |
| `n* lin *.csv`, `n* sat *.csv`, `n* Vd *.csv`, and corresponding `p*` files | Original instrument exports with measurement metadata |
| `MOSFET-data.xlsx` | Excel export of the original Google Sheets workbook, including transistor and inverter sheets |
| `SEMI208-FET.lib` | Supplied reference model text; separate from the fitted models printed by the scripts |
| `발표시연자료/` | Presentation demo Python source and the five NMOS tables it uses |
| `집적공정개론_발표시연/` | Measurement tables accompanying the original executable demo; executable excluded |
| `testdata/` | Supporting PN-junction measurements, a laboratory report, and an Excel table |

Original Korean filenames and report content are retained. This README is an English guide rather than a translation of every source comment or laboratory document.

## Setup

Use a Python environment compatible with PyTorch. A Python 3.11 or 3.12 virtual environment is a practical starting point; the original project did not provide pinned dependency versions.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

`openpyxl` is included for exploring the workbook exports. Model training uses the CSV files directly. No API key, hosted service, or pretrained checkpoint is required. Training runs on CPU by default.

## Run the experiments

From the repository root:

```bash
python MOSFET-Modeling.py
python pMOS-Modeling.py
python inv.py
```

Run one command at a time. Scripts train their models before presenting results; plotting windows may need to be closed before the next stage continues. The NMOS script subsequently prompts for either current prediction or target-power optimization. The presentation demo is launched with:

```bash
python "발표시연자료/발표용.py"
```

The three main scripts now resolve CSV paths relative to their own files, replacing the original machine-specific Google Drive path. The modeling equations, measurement tables, and training settings are otherwise preserved.

## Data conventions and architecture

The consolidated CSV files have two header rows: sweep groups such as `LIN_1`, `SAT_1`, and `Vd_1`, followed by columns such as `Gate`, `Drain`, and `Id`. They should not be treated as simple single-header tables. Raw instrument CSVs have additional setup metadata and are not direct substitutes for the consolidated files.

- Inputs: `[W, L, Vgs, Vds]`, with geometry in metres and voltages in volts.
- Training geometries: W = 10 µm; L = 1, 2, 3, 5, or 10 µm.
- Targets: `log10(Id)` for positive NMOS current, or `log10(abs(Id))` for PMOS; currents below the script's threshold are filtered.
- Transfer sweeps assume Vds = +1/+4 V for NMOS and −1/−4 V for PMOS. Output-sweep group indices map to gate voltages of magnitude 2/3/4 V.
- Network: fully connected `4 → 64 → 128 → 64 → 1` with SiLU activations, standardized inputs, Adam, and mean-squared error in log-current space.
- The standalone transistor scripts use a random 80/20 point split and train for 1,000 epochs. The inverter script trains both devices on all loaded points.
- Parameter fitting optimizes VTO, KP, LAMBDA, THETA, and ETA for 2,000 steps. Its equation is simplified; printing `level=3` does not establish equivalence to a simulator's full Level-3 model.
- The inverter example uses VDD = 4 V and balances NMOS/PMOS predicted currents over an input sweep. It includes a boundary fallback if a root cannot be bracketed.

## Limitations to consider

- No fixed random seed or dependency lock was supplied, so repeated results can differ.
- Input normalization in the standalone models is computed before the train/test split. Point-wise splits are not held-out-device validation.
- Width is constant in the training data. Predictions at other widths—including the inverter example's 20 µm PMOS—are extrapolations with no demonstrated width generalization.
- The presentation demo prints current using `10e6` as its µA scale factor; this is ten times the usual `1e6` conversion. Its original source is retained, so interpret that display accordingly.
- The reference library starts its NMOS declaration with `model` rather than `.model`; verify syntax and simulator compatibility before use.
- No trained model weights, benchmark accuracy, or reproduction of historical plots is claimed by this import.

## Import and provenance

The import preserves the course files and adds this guide, `requirements.txt`, `.gitignore`, and a file inventory. The Google Sheets workbook is included as an `.xlsx` export. The 89 files in `2025-2_Termproject.zip` were byte-identical to existing raw CSV files, so the redundant archive was omitted. Empty directories, antivirus temporary files, the packaged demo `.exe`, and duplicate ZIP bundles are omitted.

The supporting PN-junction report credits its original course team. Authorship and attribution in source documents are retained. No new open-source license is assigned to these course materials by this import; third-party and institutional rights remain with their respective owners.
