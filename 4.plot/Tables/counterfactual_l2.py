###
# Table 6: Prediction-distance L2 statistics under counterfactual terrain-label swapping.
###

import os
import glob
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    "seed": 42,
    "min_traj_len": 5,
    "embed_dim": 8,
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
# Execution & Statistics
# -----------------------
def forward_with_label(model, proprio, heightmap, label_value):
    labels = torch.full((proprio.shape[0],), label_value, dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(proprio, heightmap, labels)
    return output


def l2_per_sample(a, b):
    return torch.norm(a - b, dim=1).cpu().numpy()


def report_stats(name, arr):
    print(f"  {name:28s} mean={arr.mean():.4f}  median={np.median(arr):.4f}  sd={arr.std():.4f}  n={len(arr)}")


def main():
    X_test, y_test, terrain_to_index = load_test_data()

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

    # Flat -> Stairway (Flat data evaluated with correct flat label vs wrong stairway label)
    flat_out_correct = forward_with_label(model, flat_proprio, flat_height, flat_idx)
    flat_out_wrong = forward_with_label(model, flat_proprio, flat_height, stair_idx)
    flat_output_dist = l2_per_sample(flat_out_correct, flat_out_wrong)

    # Stairway -> Flat (Stairway data evaluated with correct stairway label vs wrong flat label)
    stair_out_correct = forward_with_label(model, stair_proprio, stair_height, stair_idx)
    stair_out_wrong = forward_with_label(model, stair_proprio, stair_height, flat_idx)
    stair_output_dist = l2_per_sample(stair_out_correct, stair_out_wrong)

    print("\n=== Table Statistics (Separate Test Set) ===")
    report_stats("Flat -> Stairway", flat_output_dist)
    report_stats("Stairway -> Flat", stair_output_dist)


if __name__ == "__main__":
    main()
