import json
import os
import re

ratios = [0.05, 0.10, 0.25, 0.50]
labels_count = {0.05: 322, 0.10: 645, 0.25: 1612, 0.50: 3225}

results = {}
for r in ratios:
    path = f"results_0.10_backup/metrics.json" if r == 0.10 else f"results_{r}/metrics.json"
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            metrics = json.load(f)
            results[r] = metrics

if len(results) < 4:
    print(f"J'ai trouvé les résultats pour {list(results.keys())}. J'attends que les 4 soient finis pour mettre à jour le rapport !")
else:
    rows = []
    for r in ratios:
        m = results[r]
        row = f"{int(r*100)}\\%  & {labels_count[r]:<4} & {m['auroc']:.4f} & {m['auprc']:.4f} & {m['f1']:.4f} & {m['ece']:.4f} \\\\"
        rows.append(row)
        
    latex_rows = "\n".join(rows)
    
    with open("rapport.tex", "r", encoding="utf-8") as f:
        content = f.read()
        
    # Remplacement spécifique pour la Table 7
    pattern = r"(\\textbf\{Ratio\} & \\textbf\{Labels \(\+\)\}.*?\\midrule\n)(.*?)(?=\n\\bottomrule)"
    
    def repl(match):
        return match.group(1) + latex_rows
        
    new_content = re.sub(pattern, repl, content, flags=re.DOTALL)
    
    with open("rapport.tex", "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print("Le rapport a été mis à jour automatiquement avec les nouvelles valeurs d'ablation !")
