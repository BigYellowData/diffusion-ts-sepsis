import numpy as np
import json

y_true = np.load("results/y_true.npy")
mean_prob = np.load("results/mean_prob.npy")
uncertainty = np.load("results/uncertainty.npy") * 1000

with open("results/metrics.json") as f:
    metrics = json.load(f)

threshold = metrics["threshold"]

# TP confiant: y=1, p > threshold, low variance
tp_mask = (y_true == 1) & (mean_prob > threshold * 2)
tp_idx = np.where(tp_mask)[0]
if len(tp_idx) > 0:
    idx = tp_idx[np.argmax(mean_prob[tp_idx])]
    print(f"TP confiant: idx={idx}, y=1, P={mean_prob[idx]:.4f}, Var={uncertainty[idx]:.4f}")

# TN confiant: y=0, p < threshold, low variance
tn_mask = (y_true == 0) & (mean_prob < threshold / 2)
tn_idx = np.where(tn_mask)[0]
if len(tn_idx) > 0:
    idx = tn_idx[np.argmin(uncertainty[tn_idx])]
    print(f"TN confiant: idx={idx}, y=0, P={mean_prob[idx]:.4f}, Var={uncertainty[idx]:.4f}")

# FP incertain: y=0, p > threshold, high variance
fp_mask = (y_true == 0) & (mean_prob > threshold * 1.5)
fp_idx = np.where(fp_mask)[0]
if len(fp_idx) > 0:
    idx = fp_idx[np.argmax(uncertainty[fp_idx])]
    print(f"FP incertain: idx={idx}, y=0, P={mean_prob[idx]:.4f}, Var={uncertainty[idx]:.4f}")

# FN silencieux: y=1, p < threshold, low variance
fn_mask = (y_true == 1) & (mean_prob < threshold / 2)
fn_idx = np.where(fn_mask)[0]
if len(fn_idx) > 0:
    idx = fn_idx[np.argmin(uncertainty[fn_idx])]
    print(f"FN silencieux: idx={idx}, y=1, P={mean_prob[idx]:.4f}, Var={uncertainty[idx]:.4f}")
