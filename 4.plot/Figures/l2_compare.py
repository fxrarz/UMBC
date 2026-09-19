###
# Figure 12: Distribution of L2 action distance between UMBC predictions and the corresponding terrain-specific reference actions.
# Figure 13. Joint-level comparison between the UMBC policy and the corresponding terrain-specific reference action.
###

import os
import glob
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))

# Import helper functions directly from utils
from scripts.UMBC.utils.data_loader import build_manifest, load_trajectories
from scripts.UMBC.utils.models import LocomotionPolicy

# -----------------------
# Config — update paths as needed
# -----------------------
CONFIG = {
    "model_path": "scripts/UMBC/logs/models/umbc.pth",
    "terrain_mapping_path": "scripts/UMBC/logs/models/terrain_mapping.json",
    "test_dirs": {
        "flat": "./scripts/UMBC/dataset/test/flat",
        "stairway": "./scripts/UMBC/dataset/test/stairway",
    },
    "output_dir": "scripts/UMBC/logs/images",
    "seed": 42,
    "min_traj_len": 5,
    "embed_dim": 8,
    "sample_index_for_fig1": 300,
}

SEED = CONFIG["seed"]
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_test_data():
    manifests = [build_manifest(t, d) for t, d in CONFIG["test_dirs"].items()]
    manifest = pd.concat(manifests, axis=0, ignore_index=True)
    manifest = manifest[manifest["length"] >= CONFIG["min_traj_len"]].reset_index(drop=True)

    full_obs_cols = [f"obs_{i}" for i in range(1, 257)]
    X_test, y_test = load_trajectories(manifest, full_obs_cols)

    if os.path.exists(CONFIG["terrain_mapping_path"]):
        with open(CONFIG["terrain_mapping_path"]) as f:
            terrain_to_index = json.load(f)
    else:
        terrain_to_index = {name: idx for idx, name in enumerate(sorted(CONFIG["test_dirs"].keys()))}
    print(f"Terrain mapping: {terrain_to_index}")

    X_test["terrain_name"] = X_test["terrain_name"].map(terrain_to_index)
    return X_test, y_test, terrain_to_index


# -----------------------
# Plotting and Execution
# -----------------------
def forward_with_label(model, proprio, heightmap, label_value):
    labels = torch.full((proprio.shape[0],), label_value, dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(proprio, heightmap, labels)
    return output

def l2_per_sample(a, b):
    return torch.norm(a - b, dim=1).cpu().numpy()

def main():
    X_test, y_test, terrain_to_index = load_test_data()
    motor_cols = y_test.columns.tolist()

    model = LocomotionPolicy(num_terrains=len(terrain_to_index), embed_dim=CONFIG["embed_dim"]).to(device)
    model.load_state_dict(torch.load(CONFIG["model_path"], weights_only=False, map_location=device))
    model.eval()

    flat_idx = terrain_to_index["flat"]
    stair_idx = terrain_to_index["stairway"]

    # Filter data and ground truth expert actions
    flat_mask = X_test["terrain_name"] == flat_idx
    stair_mask = X_test["terrain_name"] == stair_idx

    X_flat = X_test[flat_mask]
    y_flat = y_test[flat_mask]

    X_stair = X_test[stair_mask]
    y_stair = y_test[stair_mask]

    def to_tensors(df_x, df_y):
        arr_x = torch.tensor(df_x.astype(np.float32).values).to(device)
        proprio = arr_x[:, 1:70]
        heightmap = arr_x[:, 70:257]
        actions = torch.tensor(df_y.astype(np.float32).values).to(device)
        return proprio, heightmap, actions

    flat_proprio, flat_height, expert_flat_actions = to_tensors(X_flat, y_flat)
    stair_proprio, stair_height, expert_stair_actions = to_tensors(X_stair, y_stair)

    # UMBC Predictions
    umbc_flat_out = forward_with_label(model, flat_proprio, flat_height, flat_idx)
    umbc_stair_out = forward_with_label(model, stair_proprio, stair_height, stair_idx)

    # Compute action L2 distance (Prediction Error)
    flat_error_dist = l2_per_sample(umbc_flat_out, expert_flat_actions)
    stair_error_dist = l2_per_sample(umbc_stair_out, expert_stair_actions)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # -----------------------
    # Figure 1: Joint prediction comparison for a single state
    # -----------------------
    i = CONFIG["sample_index_for_fig1"]
    n_motors = len(motor_cols)
    x = np.arange(n_motors)
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].bar(x - width / 2, umbc_flat_out[i].cpu().numpy(), width=width, label="UMBC Policy")
    axes[0].bar(x + width / 2, expert_flat_actions[i].cpu().numpy(), width=width, label="Flat Ground Reference Action")
    axes[0].set_ylabel("Action value")
    axes[0].set_title(f"Flat Terrain (Index {i}): UMBC vs Ground Truth")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].legend(fontsize=10, loc='upper left')

    axes[1].bar(x - width / 2, umbc_stair_out[i].cpu().numpy(), width=width, label="UMBC Policy")
    axes[1].bar(x + width / 2, expert_stair_actions[i].cpu().numpy(), width=width, label="Stairway Reference Action")
    axes[1].set_ylabel("Action value")
    axes[1].set_title(f"Stairway Terrain (Index {i}): UMBC vs Ground Truth")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].legend(fontsize=10, loc='upper left')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(motor_cols, rotation=45, ha="right")

    fig.tight_layout()
    fig1_path = os.path.join(CONFIG["output_dir"], "14.umbc_vs_reference.png")
    fig.savefig(fig1_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved Figure 1 to {fig1_path}")
    plt.close(fig)

    # -----------------------
    # Figure 2: L2 distance distribution across the separate test set
    # -----------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.violinplot([flat_error_dist, stair_error_dist], showmeans=True, showextrema=True, widths=0.9)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Flat Error", "Stairway Error"])
    ax.set_ylabel("L2 Distance")
    ax.set_title("Action Prediction Error vs Expert Dataset")
    fig.tight_layout()
    
    fig2_path = os.path.join(CONFIG["output_dir"], "13.action_prediction_error.png")
    fig.savefig(fig2_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved Figure 2 to {fig2_path}")
    plt.close(fig)
    # -----------------------
    # Compute & Print Metrics
    # -----------------------
    print("\n" + "="*45)
    print("      L2 ACTION PREDICTION ERROR METRICS     ")
    print("="*45)
    print(f"Flat Terrain:")
    print(f"  • Mean   : {np.mean(flat_error_dist):.4f}")
    print(f"  • Median : {np.median(flat_error_dist):.4f}")
    print(f"  • Std    : {np.std(flat_error_dist):.4f}")
    print("-" * 45)
    print(f"Stairway Terrain:")
    print(f"  • Mean   : {np.mean(stair_error_dist):.4f}")
    print(f"  • Median : {np.median(stair_error_dist):.4f}")
    print(f"  • Std    : {np.std(stair_error_dist):.4f}")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()
