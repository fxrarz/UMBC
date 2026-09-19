###
# Figure 10: Distribution of prediction-distance L2 values under counterfactual terrain-label swapping.
# Figure 11. Counterfactual action comparison for selected flat-ground and stairway test states (sample index 300).
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

    X_flat = X_test[X_test["terrain_name"] == flat_idx]
    X_stair = X_test[X_test["terrain_name"] == stair_idx]

    def to_tensors(df):
        arr = torch.tensor(df.astype(np.float32).values).to(device)
        proprio = arr[:, 1:70]
        heightmap = arr[:, 70:257]
        return proprio, heightmap

    flat_proprio, flat_height = to_tensors(X_flat)
    stair_proprio, stair_height = to_tensors(X_stair)

    flat_out_correct = forward_with_label(model, flat_proprio, flat_height, flat_idx)
    flat_out_wrong = forward_with_label(model, flat_proprio, flat_height, stair_idx)

    stair_out_correct = forward_with_label(model, stair_proprio, stair_height, stair_idx)
    stair_out_wrong = forward_with_label(model, stair_proprio, stair_height, flat_idx)

    flat_output_dist = l2_per_sample(flat_out_correct, flat_out_wrong)
    stair_output_dist = l2_per_sample(stair_out_correct, stair_out_wrong)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # -----------------------
    # Figure 1: Joint prediction comparison for a single state
    # -----------------------
    i = CONFIG["sample_index_for_fig1"]
    n_motors = len(motor_cols)
    x = np.arange(n_motors)
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

    axes[0].bar(x - width / 2, flat_out_correct[i].cpu().numpy(), width=width, label="flat label (correct)")
    axes[0].bar(x + width / 2, flat_out_wrong[i].cpu().numpy(), width=width, label="stairway label (wrong)")
    axes[0].set_ylabel("Predicted action value")
    axes[0].set_title(f"Flat state (sample index {i}): correct vs swapped label")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].legend(fontsize=10, loc='upper left')

    axes[1].bar(x - width / 2, stair_out_correct[i].cpu().numpy(), width=width, label="stairway label (correct)")
    axes[1].bar(x + width / 2, stair_out_wrong[i].cpu().numpy(), width=width, label="flat label (wrong)")
    axes[1].set_ylabel("Predicted action value")
    axes[1].set_title(f"Stairway state (sample index {i}): correct vs swapped label")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].legend(fontsize=10, loc='upper left')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(motor_cols, rotation=45, ha="right")

    fig.tight_layout()
    fig1_path = os.path.join(CONFIG["output_dir"], "11.cross_label_test.png")
    fig.savefig(fig1_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved Figure 1 to {fig1_path}")
    plt.close(fig)

    # -----------------------
    # Figure 2: L2 distance distribution across the separate test set
    # -----------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.violinplot([flat_output_dist, stair_output_dist], showmeans=True, showextrema=True, widths=0.9)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["flat", "stairway"])
    ax.set_ylabel("L2 distance")
    ax.set_title("Predicted-action distance: correct vs swapped terrain label")
    fig.tight_layout()
    
    fig2_path = os.path.join(CONFIG["output_dir"], "10.action_l2_distance_distribution_test.png")
    fig.savefig(fig2_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved Figure 2 to {fig2_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
