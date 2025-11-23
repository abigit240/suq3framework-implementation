import os
import pandas as pd
import matplotlib.pyplot as plt
import json

# Paths
RESULTS_CSV = './results/pipeline_results.csv'
HISTORIES_DIR = './results/histories/'  # assumes JSON histories saved per stage
PLOTS_DIR = './results/'
RESULTS_DIR = './results/'

os.makedirs(PLOTS_DIR, exist_ok=True)

# Load pipeline results
df_results = pd.read_csv(RESULTS_CSV)
print("Pipeline Results:")
print(df_results)

# ------------------------
# Accuracy & F1 Score Comparison
# ------------------------
plt.figure(figsize=(10, 6))
bar_width = 0.35
x = range(len(df_results))
plt.bar(x, df_results['Accuracy'], width=bar_width, alpha=0.7, label='Accuracy')
plt.bar([i + bar_width for i in x], df_results['F1'], width=bar_width, alpha=0.7, label='F1 Score')
plt.xticks([i + bar_width/2 for i in x], df_results['Stage'])
plt.ylabel('Score')
plt.title('Accuracy and F1 Score Comparison Across Stages')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_f1_comparison.png'), dpi=300)
plt.close()

# ------------------------
# Model Size vs Sparsity
# ------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df_results['Sparsity'], df_results['Size (KB)'], s=100, c='r')
for i, txt in enumerate(df_results['Stage']):
    plt.annotate(txt, (df_results['Sparsity'][i], df_results['Size (KB)'][i]), textcoords="offset points", xytext=(5,5), ha='center')
plt.xlabel('Sparsity')
plt.ylabel('Model Size (KB)')
plt.title('Model Size vs Sparsity')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'size_vs_sparsity.png'), dpi=300)
plt.close()

# ------------------------
# Loss Curves Across Stages (Separate Subplots)
# ------------------------
stage_files = [f for f in os.listdir(HISTORIES_DIR) if f.endswith('.json')]
if stage_files:
    plt.figure(figsize=(16, 12))
    for idx, stage_file in enumerate(sorted(stage_files)):
        stage_name = stage_file.replace('.json', '')
        with open(os.path.join(HISTORIES_DIR, stage_file), 'r') as f:
            history = json.load(f)
        
        if 'loss' not in history or 'val_loss' not in history:
            print(f"Skipping {stage_name}, missing 'loss' or 'val_loss'")
            continue

        plt.subplot(2, 2, idx+1)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], '--', label='Val Loss')
        plt.title(f'{stage_name.capitalize()} Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_curves_all_stages.png'), dpi=300)
    plt.close()
else:
    print("No history files found for loss curves.")

# ------------------------
# Save summary results to JSON
# ------------------------
summary_json_path = os.path.join(RESULTS_DIR, 'summary_results.json')
df_results.to_json(summary_json_path, orient='records', indent=4)
print(f"Summary results saved to {summary_json_path}")
print(f"All plots saved to {PLOTS_DIR}")
