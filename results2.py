import os
import json
import matplotlib.pyplot as plt

HISTORIES_DIR = './results/histories/'  # JSON histories per stage
PLOTS_DIR = './results/'
os.makedirs(PLOTS_DIR, exist_ok=True)

stages = [
    "Baseline",
    "Stage 1 (Structured)",
    "Stage 2 (Unstructured)",
    "Stage 3 (QAT+TFLite)"
]

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

found_any = False

for i, stage_name in enumerate(stages):
    history_file = os.path.join(HISTORIES_DIR, f"{stage_name}.json")
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        axs[i].plot(history['loss'], label='Train Loss', color='blue')
        axs[i].plot(history['val_loss'], label='Val Loss', color='orange', linestyle='--')
        axs[i].set_title(stage_name)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('Loss')
        axs[i].grid(True, linestyle='--', alpha=0.5)
        axs[i].legend()
        found_any = True
    else:
        axs[i].text(0.5, 0.5, f"No history found\nfor {stage_name}",
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=12)
        axs[i].set_title(stage_name)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('Loss')
        axs[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'loss_curves_4_stages.png'), dpi=300)
plt.close()

if found_any:
    print(f"4-stage loss curves saved to {os.path.join(PLOTS_DIR, 'loss_curves_4_stages.png')}")
else:
    print("No history files found for any stage. Cannot plot loss curves.")
