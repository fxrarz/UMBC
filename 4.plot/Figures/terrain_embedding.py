###
# Figure 8. Terrain embedding weights.
###

import os
import glob
import re
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))

# Import helper functions directly from utils
from scripts.UMBC.utils.data_loader import build_manifest, load_trajectories
from scripts.UMBC.utils.models import LocomotionPolicy

# -----------------------
# Global config
# -----------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MIN_TRAJ_LEN = 5

if __name__ == '__main__':
    # Define paths
    target_seed = 42
    mapping_path = f"scripts/UMBC/logs/models/terrain_mapping.json"
    weights_path = f"scripts/UMBC/logs/models/umbc.pth"
    os.makedirs("scripts/UMBC/logs/images", exist_ok=True)

    terrain_dirs = {
        'flat': './scripts/UMBC/dataset/test/flat',
        'stairway': './scripts/UMBC/dataset/test/stairway',
    }

    # Load Terrain Mapping from JSON
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            terrain_to_index = json.load(f)
    else:
        print(f"Warning: {mapping_path} not found. Defaulting mapping.")
        terrain_to_index = {name: idx for idx, name in enumerate(sorted(terrain_dirs.keys()))}
        
    index_to_terrain = {v: k for k, v in terrain_to_index.items()}
    num_terrains = len(terrain_to_index)

    # 1. Get Embeddings for Untrained Model
    model_nolearn = LocomotionPolicy(num_terrains=num_terrains, embed_dim=8).to(device)
    model_nolearn.eval()
    with torch.no_grad():
        embeddings_nolearn = model_nolearn.terrain_embed.weight.cpu().numpy()

    # 2. Get Embeddings for Trained Model
    model_learn = LocomotionPolicy(num_terrains=num_terrains, embed_dim=8).to(device)
    model_learn.load_state_dict(torch.load(weights_path, weights_only=False, map_location=device))
    model_learn.eval()
    with torch.no_grad():
        embeddings_learn = model_learn.terrain_embed.weight.cpu().numpy()

    print("Embed no learn")
    print(embeddings_nolearn)
    print("Embed learn")
    print(embeddings_learn)

    # -----------------------
    # Side-by-Side Embedding Plot with Separate Colorbars
    # -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    # Left plot: Before Learning
    im0 = axes[0].imshow(embeddings_nolearn, aspect='auto', cmap='viridis')
    axes[0].set_title("Before Learning (Random Weights)")
    axes[0].set_xlabel("Embedding Dimension")
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: int(x + 1)))
    labels = [f"{index_to_terrain[i]} ({i})" for i in range(len(index_to_terrain))]
    axes[0].set_yticks(ticks=range(len(index_to_terrain)))
    axes[0].set_yticklabels(labels)
    fig.colorbar(im0, ax=axes[0], label='Latent Value', fraction=0.046, pad=0.04)

    # Right plot: After Learning
    im1 = axes[1].imshow(embeddings_learn, aspect='auto', cmap='viridis')
    axes[1].set_title("After Learning (Trained)")
    axes[1].set_xlabel("Embedding Dimension")
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: int(x + 1)))
    fig.colorbar(im1, ax=axes[1], label='Latent Value', fraction=0.046, pad=0.04)

    plt.tight_layout()
    combined_save_path = "scripts/UMBC/logs/images/8.embed.png"
    plt.savefig(combined_save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Combined embedding comparison plot saved to {combined_save_path}")

    # 3. Load Test Data for t-SNE (kept from your original script)
    print("Loading test data for t-SNE...")
    manifests = [build_manifest(terrain, data_dir) for terrain, data_dir in terrain_dirs.items()]
    manifest = pd.concat(manifests, axis=0, ignore_index=True)
    manifest = manifest[manifest["length"] >= MIN_TRAJ_LEN].reset_index(drop=True)

    _, traj_test = train_test_split(manifest, test_size=0.1, stratify=manifest["terrain"], random_state=SEED)
    
    full_obs_cols = [f"obs_{i}" for i in range(1, 257)]
    X_test, y_test = load_trajectories(traj_test, full_obs_cols)
    X_test["terrain_name"] = X_test["terrain_name"].map(terrain_to_index)

    X_test_t = torch.tensor(X_test.astype(np.float32).values).to(device)
    y_test_t = torch.tensor(y_test.astype(np.float32).values).to(device)

    test_ds = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
