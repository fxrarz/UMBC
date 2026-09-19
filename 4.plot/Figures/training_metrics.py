###
# Figure 3: Overall Loss & MAE.
# Figure 4: Action RMSE & lTerrain-Specific Loss.
###

import pickle
import matplotlib.pyplot as plt

# Load the history
with open("./scripts/UMBC/logs/metrics/umbc_history.pkl", "rb") as f:
    history = pickle.load(f)

# Dynamically set the range to match the actual number of trained epochs
actual_epochs = len(history['train_loss'])
epochs_range = range(1, actual_epochs + 1)

# Define variables to avoid NameErrors
run_name = "UMBC Policy"
save_path = "umbc"

plt.rcParams.update({
    'font.size': 18,          # General text / tick labels
    'axes.labelsize': 18,     # X and Y axis labels
    'axes.titlesize': 18,     # Plot title
    'legend.fontsize': 18,    # Legend text
    'figure.autolayout': True
})

# --- Plotting ---

# 1st Graph: Train Loss vs Val Loss
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
plt.plot(epochs_range, history['val_loss'], 'r--', label='Val Loss')
plt.title(f'Overall Loss - {run_name}')
plt.xlabel('Epochs')
plt.ylabel('Huber Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = f"./scripts/UMBC/logs/images/{save_path}_overall_loss.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# 2nd Graph: Val Loss vs Flat Loss vs Stairway Loss
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, history['val_loss'], 'k-', alpha=0.3, label='Total Val')
plt.plot(epochs_range, history['val_flat_loss'], 'g--', label='Flat Ground')
# Fixed the dictionary key to match the terrain name 'stairway'
plt.plot(epochs_range, history['val_stairway_loss'], 'r--', label='Stairway')
plt.title(f'Terrain-Specific Loss - {run_name}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plot_path = f"./scripts/UMBC/logs/images/{save_path}_terrain_breakdown.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# 3rd Graph: Train MAE vs Val MAE
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, history['train_mae'], 'b-', label='Train MAE')
plt.plot(epochs_range, history['val_mae'], 'orange', linestyle='--', label='Val MAE')
plt.title(f'Precision (MAE) - {run_name}')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plot_path = f"./scripts/UMBC/logs/images/{save_path}_mae_precision.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# 4th Graph: Validation RMSE
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, history['val_rmse'], 'm-', linewidth=2, label='Val RMSE')
plt.title(f'Action RMSE - {run_name}')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plot_path = f"./scripts/UMBC/logs/images/{save_path}_rmse_precision.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

plt.show()
