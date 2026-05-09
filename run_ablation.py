import os
import shutil
import subprocess

ratios = [0.05, 0.25, 0.50]

print("=== DEBUT DE L'ETUDE D'ABLATION ===")

# Backup du dossier results actuel (le 10%)
if os.path.exists("results"):
    if not os.path.exists("results_0.10_backup"):
        print("Sauvegarde du modèle 10%...")
        shutil.copytree("results", "results_0.10_backup")
        
for r in ratios:
    print(f"\n--- Lancement pour label_ratio = {r} ---")
    
    # Lancer uniquement le classifieur et l'évaluation
    cmd1 = f"uv run main.py --stage classifier --label_ratio {r}"
    subprocess.run(cmd1, shell=True)
    cmd2 = f"uv run main.py --stage evaluate --label_ratio {r}"
    subprocess.run(cmd2, shell=True)
    
    # Sauvegarder les résultats spécifiques
    res_dir = f"results_{r}"
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    
    shutil.copytree("results", res_dir)
    print(f"Résultats pour {r} sauvegardés dans {res_dir}")

print("\n=== FIN DE L'ETUDE D'ABLATION ===")
print("Vos résultats initiaux (10%) sont conservés dans results_0.10_backup")
