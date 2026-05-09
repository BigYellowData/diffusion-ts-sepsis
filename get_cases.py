import numpy as np
import json

y_true = np.load("results/y_true.npy")
mean_prob = np.load("results/mean_prob.npy")
uncertainty = np.load("results/uncertainty.npy")

with open("results/metrics.json") as f:
    metrics = json.load(f)

threshold = metrics["threshold"]

test_idx = [153, 5100, 3991, 14008]
cases = ["TP confiant", "TN confiant", "FP incertain", "FN silencieux"]

print(f"{'Cas':<15} {'y_vrai'} {'P(sepsis)':<10} {'Var MC (x10^-3)':<15} {'Décision'}")
print("-" * 65)

for name, idx in zip(cases, test_idx):
    y = y_true[idx]
    p = mean_prob[idx]
    u = uncertainty[idx] * 1000  # x10^-3
    
    if p >= threshold:
        decision = "SEPSIS " + ("V" if y == 1 else "X")
    else:
        decision = "non-sepsis " + ("V" if y == 0 else "X")
        
    print(f"{name:<15} {int(y):<6} {p:<10.4f} {u:<15.4f} {decision}")
