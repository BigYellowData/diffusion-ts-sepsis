import numpy as np
import json
from src.data.preprocess import FEATURE_COLS

data = np.load("data/processed/splits.npz", allow_pickle=True)
splits = data['splits'].item()
scaler = data['scaler'].item()

y_true = np.load("results/y_true.npy")
mean_prob = np.load("results/mean_prob.npy")
uncertainty = np.load("results/uncertainty.npy") * 1000

with open("results/metrics.json") as f:
    metrics = json.load(f)

threshold = metrics["threshold"]

test_idx = [3351, 14263, 4141, 2313]
cases = ["TP confiant", "TN confiant", "FP incertain", "FN silencieux"]

X_test = splits['test']['X'][test_idx]

print("Cas | HR | Temp | MAP | Lactate | P(sepsis) | Var MC")
print("-" * 60)

for i, (name, idx) in enumerate(zip(cases, test_idx)):
    X = X_test[i] * scaler.scale_ + scaler.mean_
    
    hr_min, hr_max = X[:, FEATURE_COLS.index('HR')].min(), X[:, FEATURE_COLS.index('HR')].max()
    temp_min, temp_max = X[:, FEATURE_COLS.index('Temp')].min(), X[:, FEATURE_COLS.index('Temp')].max()
    map_min, map_max = X[:, FEATURE_COLS.index('MAP')].min(), X[:, FEATURE_COLS.index('MAP')].max()
    lac_min, lac_max = X[:, FEATURE_COLS.index('Lactate')].min(), X[:, FEATURE_COLS.index('Lactate')].max()
    
    # Handle NaNs if variable wasn't measured (using normalized value 0 -> mean)
    # Wait, in PhysioNet NaNs are imputed, but let's just print min-max
    p = mean_prob[idx]
    u = uncertainty[idx]
    
    print(f"{name} | {hr_min:.0f}-{hr_max:.0f} | {temp_min:.1f}-{temp_max:.1f} | {map_min:.0f}-{map_max:.0f} | {lac_min:.1f}-{lac_max:.1f} | {p:.4f} | {u:.4f}")
