import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import glob
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt

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
# Data Loading Functions
# -----------------------
def build_manifest(terrain, data_dir):
    manifest_path = os.path.join(data_dir, "manifest.csv")
    if os.path.exists(manifest_path):
        m = pd.read_csv(manifest_path)
        m["terrain"] = terrain
        m["obs_path"] = m["obs_file"].apply(lambda f: os.path.join(data_dir, f))
        m["action_path"] = m["action_file"].apply(lambda f: os.path.join(data_dir, f))
        m["traj_key"] = terrain + "_" + m["env_id"].astype(str) + "_" + m["traj_idx"].astype(str)
        return m[["terrain", "traj_key", "obs_path", "action_path", "length"]]

    rows = []
    for obs_path in sorted(glob.glob(os.path.join(data_dir, "traj_*_obs_df.csv"))):
        match = re.search(r"traj_(\d+)_obs_df\.csv$", obs_path)
        if not match:
            continue
        idx = match.group(1)
        action_path = os.path.join(data_dir, f"traj_{idx}_action_df.csv")
        if not os.path.exists(action_path):
            continue
        length = len(pd.read_csv(obs_path))
        rows.append({
            "terrain": terrain,
            "traj_key": f"{terrain}_{idx}",
            "obs_path": obs_path,
            "action_path": action_path,
            "length": length,
        })
    return pd.DataFrame(rows)

def load_trajectories(manifest_subset, full_obs_cols):
    obs_frames, action_frames = [], []
    for _, row in manifest_subset.iterrows():
        obs_df = pd.read_csv(row["obs_path"])
        act_df = pd.read_csv(row["action_path"])
        n = min(len(obs_df), len(act_df))
        obs_df = obs_df.iloc[:n].copy()
        act_df = act_df.iloc[:n].copy()
        obs_df = obs_df.drop(columns=["step"], errors="ignore")
        obs_df["terrain_name"] = row["terrain"]
        obs_frames.append(obs_df)
        action_frames.append(act_df.drop(columns=["step"], errors="ignore"))

    observation_df = pd.concat(obs_frames, axis=0, ignore_index=True)
    action_df = pd.concat(action_frames, axis=0, ignore_index=True)

    observation_df = observation_df.reindex(columns=["terrain_name"] + full_obs_cols, fill_value=0)
    observation_df.fillna(0, inplace=True)
    return observation_df, action_df

# -----------------------
# Model Architecture
# -----------------------
class UMBC(nn.Module):
    def __init__(self, proprio_dim=69, heightmap_dim=187, output_dim=19, num_terrains=2, embed_dim=8):
        super().__init__()
        self.proprio_net = nn.Sequential(
            nn.Linear(proprio_dim, 256),
            nn.LeakyReLU(0.5),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.perception_net = nn.Sequential(
            nn.Linear(heightmap_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.terrain_embed = nn.Embedding(num_terrains, embed_dim)
        self.final_head = nn.Sequential(
            nn.Linear(128 + 64 + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, proprio, heightmap, terrain_label):
        h_ordered = heightmap.view(-1, 11, 17).transpose(1, 2).flatten(1)
        z_p = self.proprio_net(proprio)
        z_h = self.perception_net(h_ordered)
        z_l = self.terrain_embed(terrain_label.long())
        combined = torch.cat((z_p, z_h, z_l), dim=1)
        return self.final_head(combined)

class OneHot(nn.Module):
    def __init__(self, proprio_dim=69, heightmap_dim=187, output_dim=19, num_terrains=2, embed_dim=8):
        super().__init__()
        self.num_terrains = num_terrains
        self.proprio_net = nn.Sequential(
            nn.Linear(proprio_dim, 256),
            nn.LeakyReLU(0.5),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.perception_net = nn.Sequential(
            nn.Linear(heightmap_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.final_head = nn.Sequential(
            nn.Linear(128 + 64 + num_terrains, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, proprio, heightmap, terrain_label):
        h_ordered = heightmap.view(-1, 11, 17).transpose(1, 2).flatten(1)
        z_p = self.proprio_net(proprio)
        z_h = self.perception_net(h_ordered)
        z_l = F.one_hot(terrain_label.long(), num_classes=self.num_terrains).float()
        combined = torch.cat((z_p, z_h, z_l), dim=1)
        return self.final_head(combined)

class NoLabel(nn.Module):
    def __init__(self, proprio_dim=69, heightmap_dim=187, output_dim=19, num_terrains=2, embed_dim=0):
        super().__init__()
        self.proprio_net = nn.Sequential(
            nn.Linear(proprio_dim, 256),
            nn.LeakyReLU(0.5),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.perception_net = nn.Sequential(
            nn.Linear(heightmap_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.final_head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, proprio, heightmap, terrain_label):
        h_ordered = heightmap.view(-1, 11, 17).transpose(1, 2).flatten(1)
        z_p = self.proprio_net(proprio)
        z_h = self.perception_net(h_ordered)
        combined = torch.cat((z_p, z_h), dim=1)
        return self.final_head(combined)

class Concat(nn.Module):
    def __init__(self, proprio_dim=69, heightmap_dim=187, output_dim=19, num_terrains=2, embed_dim=8):
        super().__init__()
        self.terrain_embed = nn.Embedding(num_terrains, embed_dim)
        self.final_head = nn.Sequential(
            nn.Linear(proprio_dim + heightmap_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, proprio, heightmap, terrain_label):
        h_ordered = heightmap.view(-1, 11, 17).transpose(1, 2).flatten(1)
        z_l = self.terrain_embed(terrain_label.long())
        combined = torch.cat((proprio, h_ordered, z_l), dim=1)
        return self.final_head(combined)

def main(weight_name, network=UMBC):
    test_dirs = {
        'flat': './scripts/UMBC/dataset/test/flat',
        'stairway': './scripts/UMBC/dataset/test/stairway',
    }

    # 1. Load the pre-saved terrain mapping
    mapping_path = "./scripts/UMBC/logs/models/terrain_mapping.json"
    with open(mapping_path, "r") as f:
        terrain_to_index = json.load(f)
    num_terrains = len(terrain_to_index)
    print(f"Loaded terrain mapping: {terrain_to_index}")

    # 2. Build the test manifest
    manifests = [build_manifest(terrain, data_dir) for terrain, data_dir in test_dirs.items()]
    manifest = pd.concat(manifests, axis=0, ignore_index=True)
    manifest = manifest[manifest["length"] >= MIN_TRAJ_LEN].reset_index(drop=True)
    print(f"Loaded {len(manifest)} test trajectories.")

    # 3. Load the raw test data into memory
    full_obs_cols = [f"obs_{i}" for i in range(1, 257)]
    X_test_df, y_test_df = load_trajectories(manifest, full_obs_cols)
    X_test_df["terrain_name"] = X_test_df["terrain_name"].map(terrain_to_index)

    X_t = torch.tensor(X_test_df.astype(np.float32).values).to(device)
    y_t = torch.tensor(y_test_df.astype(np.float32).values).to(device)

    # 4. Initialize model and load weights
    model = network(num_terrains=num_terrains, embed_dim=8).to(device)
    weights_path = f"./scripts/UMBC/logs/models/{weight_name}.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 5. Run Inference on the entire test set & Calculate Metrics
    with torch.no_grad():
        label_input = X_t[:, 0].long()
        proprio_input = X_t[:, 1:70]
        heightmap_input = X_t[:, 70:257]
        
        preds = model(proprio_input, heightmap_input, label_input)
        
        # Overall metrics
        sq_err = (preds - y_t) ** 2
        abs_err = torch.abs(preds - y_t)
        
        overall_rmse = torch.sqrt(sq_err.mean()).item()
        overall_mae = abs_err.mean().item()
        
        print(f"\n=====================================")
        print(f"OVERALL TEST METRICS:")
        print(f"  RMSE: {overall_rmse:.6f}")
        print(f"  MAE:  {overall_mae:.6f}")
        print(f"=====================================\n")
        
        # Per-terrain metrics (Flat, Stairway, etc.)
        print("PER-TERRAIN TEST METRICS:")
        for terrain_name, terrain_idx in terrain_to_index.items():
            mask = (label_input == terrain_idx)
            if mask.sum() > 0:
                t_preds = preds[mask]
                t_y = y_t[mask]
                
                t_sq_err = (t_preds - t_y) ** 2
                t_abs_err = torch.abs(t_preds - t_y)
                
                t_rmse = torch.sqrt(t_sq_err.mean()).item()
                t_mae = t_abs_err.mean().item()
                
                print(f"  -> {terrain_name.capitalize()}:")
                print(f"     RMSE: {t_rmse:.6f}")
                print(f"     MAE:  {t_mae:.6f}")
        print(f"=====================================\n")

weight_name = "umbc"
main(weight_name, UMBC)

weight_name = "umbc_seed_123"
main(weight_name, UMBC)

weight_name = "umbc_seed_456"
main(weight_name, UMBC)

weight_name = "umbc_seed_789"
main(weight_name, UMBC)

weight_name = "umbc_seed_999"
main(weight_name, UMBC)

weight_name = "adaptive_weight"
main(weight_name, UMBC)

weight_name = "1hot"
main(weight_name, OneHot)

weight_name = "nolabel"
main(weight_name, NoLabel)

weight_name = "concat"
main(weight_name, Concat)

