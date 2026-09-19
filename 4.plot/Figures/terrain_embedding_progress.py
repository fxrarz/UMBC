###
# Figure 9. Epoch-wise evolution of the 8-dimensional terrain embedding
###

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -----------------------
# Global config
# -----------------------
target_seed = 42
model_name = f"umbc"
history_path = f"scripts/UMBC/logs/metrics/{model_name}_history.pkl"
mapping_path = f"scripts/UMBC/logs/models/terrain_mapping.json"
save_path = "scripts/UMBC/logs/images/embedding_progression.png"

def plot_embedding_progression():
    # 1. Load Terrain Mapping
    if not os.path.exists(mapping_path):
        print(f"Error: Mapping file {mapping_path} not found.")
        return
        
    with open(mapping_path, "r") as f:
        terrain_to_index = json.load(f)
    index_to_terrain = {v: k for k, v in terrain_to_index.items()}
    num_terrains = len(index_to_terrain)

    # 2. Load Training History
    if not os.path.exists(history_path):
        print(f"Error: History file {history_path} not found. Did you retrain with embedding logging?")
        return

    with open(history_path, "rb") as f:
        history = pickle.load(f)

    if 'embeddings' not in history or len(history['embeddings']) == 0:
        print("Error: 'embeddings' not found in history. Ensure you added the logging step in train.py.")
        return

    # Convert list of arrays to a single 3D numpy array
    # Shape will be: (num_epochs, num_terrains, embed_dim)
    embeddings_over_time = np.array(history['embeddings'])
    print("Shape will be: (num_epochs, num_terrains, embed_dim)")
    print(embeddings_over_time)
    num_epochs, _, embed_dim = embeddings_over_time.shape

    # 3. Create Side-by-Side Heatmaps
    fig, axes = plt.subplots(1, num_terrains, figsize=(6 * num_terrains, 6), sharey=True)
    
    # Handle case where there's only one terrain
    if num_terrains == 1:
        axes = [axes]

    # Find global min and max for consistent color mapping across subplots
    vmin, vmax = embeddings_over_time.min(), embeddings_over_time.max()

    for t_idx in range(num_terrains):
        ax = axes[t_idx]
        
        # Extract embeddings for this specific terrain across all epochs
        # Shape: (num_epochs, embed_dim)
        terrain_embs = embeddings_over_time[:, t_idx, :]
        
        im = ax.imshow(
            terrain_embs, 
            aspect='auto', 
            cmap='viridis', 
            interpolation='nearest',
            vmin=vmin, 
            vmax=vmax
        )
        
        ax.set_title(f"{index_to_terrain[t_idx].capitalize()} Terrain")
        ax.set_xlabel("Embedding Dimension (1-8)")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: int(x + 1)))
        
        # Only set Y-label for the leftmost plot
        if t_idx == 0:
            ax.set_ylabel("Training Epoch")
            
        # Add X-axis ticks to match embedding dimensions
        ax.set_xticks(range(embed_dim))

    # Add a shared colorbar on the right
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Latent Weight Value')

    plt.suptitle("Progression of Terrain Embeddings Over Epochs", fontsize=14, y=0.98)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Progression plot successfully saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    plot_embedding_progression()
