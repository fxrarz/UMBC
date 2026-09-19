###
# Figure 7. t-SNE projection of the 200-dimensional fused latent representation.
###

import os
import glob
import re
import json
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.cm as cm
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

# -----------------------
# Helper to extract latents
# -----------------------
def extract_latents(model, test_loader):
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i > 15: break  # Use ~2000 samples for a clean, fast plot
            
            label_in = inputs[:, 0].long()
            proprio_in = inputs[:, 1:70]
            height_in = inputs[:, 70:257]

            z = model.get_latent(proprio_in, height_in, label_in)
            latents.append(z.cpu().numpy())
            labels.append(label_in.cpu().numpy())

    X_latent = np.concatenate(latents)
    y_labels = np.concatenate(labels)

    print("Running PCA -> t-SNE...")
    pca = PCA(n_components=min(50, X_latent.shape[1]))
    X_pca = pca.fit_transform(X_latent)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    return tsne.fit_transform(X_pca), y_labels


if __name__ == '__main__':
    # Configuration
    target_seed = 42
    model_name = f"umbc"
    mapping_path = f"scripts/UMBC/logs/models/{model_name}_terrain_mapping.json"
    weights_path = f"scripts/UMBC/logs/models/{model_name}.pth"
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
        print(f"Warning: {mapping_path} not found. Defaulting to alphabetical mapping.")
        terrain_to_index = {name: idx for idx, name in enumerate(sorted(terrain_dirs.keys()))}
        
    index_to_terrain = {v: k for k, v in terrain_to_index.items()}
    num_terrains = len(terrain_to_index)
    print(f"Terrain mapping loaded: {terrain_to_index}")

    # Load Data via Manifest
    print("Loading test dataset...")
    manifests = []
    for t_name in terrain_to_index.keys():
        if t_name in terrain_dirs:
            manifests.append(build_manifest(t_name, terrain_dirs[t_name]))
            
    manifest = pd.concat(manifests, axis=0, ignore_index=True)
    manifest = manifest[manifest["length"] >= MIN_TRAJ_LEN].reset_index(drop=True)

    # Replicate exact test split used in training
    _, traj_test = train_test_split(manifest, test_size=0.1, stratify=manifest["terrain"], random_state=SEED)
    
    full_obs_cols = [f"obs_{i}" for i in range(1, 257)]
    X_test, y_test = load_trajectories(traj_test, full_obs_cols)
    X_test["terrain_name"] = X_test["terrain_name"].map(terrain_to_index)

    X_test_t = torch.tensor(X_test.astype(np.float32).values).to(device)
    y_test_t = torch.tensor(y_test.astype(np.float32).values).to(device)

    test_ds = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # 1. Get Latent Space BEFORE loading trained weights
    model = LocomotionPolicy(num_terrains=num_terrains, embed_dim=8).to(device)
    X_2d_nolearn, y_labels_nolearn = extract_latents(model, test_loader)

    # 2. Get Latent Space AFTER loading trained weights
    print(f"Loading trained weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, weights_only=False, map_location=device))
    X_2d_learn, y_labels_learn = extract_latents(model, test_loader)

    # -----------------------
    # Side-by-Side Plotting
    # -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    cmap = cm.get_cmap('tab10')

    # Left plot: Before Learning
    for t_idx in range(len(index_to_terrain)):
        indices = y_labels_nolearn == t_idx
        terrain_name = index_to_terrain[t_idx].capitalize()
        axes[0].scatter(
            X_2d_nolearn[indices, 0], 
            X_2d_nolearn[indices, 1], 
            color=cmap(t_idx), 
            label=terrain_name, 
            alpha=0.6, 
            edgecolors='w'
        )
    axes[0].set_title("Before Learning (Random Weights)")
    axes[0].set_xlabel("TSNE Dimension 1")
    axes[0].set_ylabel("TSNE Dimension 2")
    axes[0].grid(True, alpha=0.3)

    # Right plot: After Learning
    for t_idx in range(len(index_to_terrain)):
        indices = y_labels_learn == t_idx
        terrain_name = index_to_terrain[t_idx].capitalize()
        axes[1].scatter(
            X_2d_learn[indices, 0], 
            X_2d_learn[indices, 1], 
            color=cmap(t_idx), 
            label=terrain_name, 
            alpha=0.6, 
            edgecolors='w'
        )
    axes[1].set_title("After Learning (Trained)")
    axes[1].set_xlabel("TSNE Dimension 1")
    axes[1].grid(True, alpha=0.3)

    # Global legend at the top center across subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=len(index_to_terrain), fontsize=15, frameon=True)

    plt.tight_layout()
    combined_save_path = "scripts/UMBC/logs/images/7.tsne_comparison.png"
    plt.savefig(combined_save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Combined side-by-side latent plot saved to {combined_save_path}")
