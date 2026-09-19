###
# Table 3: Statistics of the final trajectory-based demonstration dataset.
###

import argparse
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))

# Import helper functions directly from utils
from scripts.UMBC.utils.data_loader import build_manifest, load_trajectories

parser = argparse.ArgumentParser(
    description="This script prints the summary of UMBC dataset."
)

parser.add_argument("--type", type=str, default="train",
                    choices=["train", "test"],
                    help="Select Train or Test")

args_cli = parser.parse_args()

if args_cli.type == "train":
    terrain_dirs = {
        'flat': './scripts/UMBC/dataset/flat',
        'stairway': './scripts/UMBC/dataset/stairway',
    }
else:
    terrain_dirs = {
        'flat': './scripts/UMBC/dataset/test/flat',
        'stairway': './scripts/UMBC/dataset/test/stairway',
    }

full_obs_cols = [f"obs_{i}" for i in range(1, 257)]

print("=" * 60)
if args_cli.type == "train":
    print("1. DATASET SIZE SUMMARY - TRAIN")
else:
    print("1. DATASET SIZE SUMMARY - TEST")
print("=" * 60)

for terrain, data_dir in terrain_dirs.items():
    manifest = build_manifest(terrain, data_dir)
    total_trajs = len(manifest)
    total_steps = manifest['length'].sum()
    mean_len = manifest['length'].mean()
    std_len = manifest['length'].std()
    
    print(f"[{terrain.upper()}]")
    print(f"  - Total Trajectories : {total_trajs:,}")
    print(f"  - Total Samples/Steps: {total_steps:,}")
    print(f"  - Trajectory Length  : {mean_len:.2f} ± {std_len:.2f} steps\n")

print("=" * 60)
print("2. COMPUTING OBSERVATION & ACTION STANDARD DEVIATION")
print("=" * 60)

# Load combined dataset via load_trajectories
all_manifests = [build_manifest(terrain, data_dir) for terrain, data_dir in terrain_dirs.items()]
combined_manifest = pd.concat(all_manifests, axis=0, ignore_index=True)

obs_df, action_df = load_trajectories(combined_manifest, full_obs_cols)

# Separate Proprioception (obs_1 to obs_69) and Height Scan (obs_70 to obs_256)
proprio_cols = [f"obs_{i}" for i in range(1, 70)]
heightmap_cols = [f"obs_{i}" for i in range(70, 257)]

proprio_data = obs_df[proprio_cols].values
height_data = obs_df[heightmap_cols].values
action_data = action_df.values

print("\n--- OVERALL FEATURE STANDARD DEVIATIONS ---")
print(f"Proprioception SD (mean across 69 features): {np.std(proprio_data, axis=0).mean():.4f}")
print(f"Heightmap Scan SD (mean across 187 features): {np.std(height_data, axis=0).mean():.4f}")
print(f"Actions SD       (mean across 19 motors):    {np.std(action_data, axis=0).mean():.4f}")

print("\n--- PER-MOTOR ACTION STANDARD DEVIATIONS ---")
for col, sd in zip(action_df.columns, np.std(action_data, axis=0)):
    print(f"  {col:10s} SD: {sd:.6f}")
