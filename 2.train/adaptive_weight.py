# python3 scripts/UMBC/2.train/adaptive_weight.py

import os
import glob
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import json
import pickle  # <-- Added import for saving history
from torch.utils.data import TensorDataset, DataLoader, BatchSampler, SubsetRandomSampler
from sklearn.model_selection import train_test_split

# -----------------------
# Global config
# -----------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# drop degenerate trajectories (e.g. envs that terminate on step 1 due to a bad reset)
MIN_TRAJ_LEN = 5


def build_manifest(terrain, data_dir):
    """Return a DataFrame with one row per trajectory: terrain, traj_key, obs_path, action_path, length."""
    manifest_path = os.path.join(data_dir, "manifest.csv")
    if os.path.exists(manifest_path):
        m = pd.read_csv(manifest_path)
        m["terrain"] = terrain
        m["obs_path"] = m["obs_file"].apply(lambda f: os.path.join(data_dir, f))
        m["action_path"] = m["action_file"].apply(lambda f: os.path.join(data_dir, f))
        m["traj_key"] = terrain + "_" + m["env_id"].astype(str) + "_" + m["traj_idx"].astype(str)
        return m[["terrain", "traj_key", "obs_path", "action_path", "length"]]

    # fallback for older-format collections without manifest.csv
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
    """Concatenate every trajectory listed in manifest_subset into one (obs_df, action_df) pair.
    full_obs_cols enforces a single, consistent column layout across terrains with different
    observation dimensionality (e.g. flat has no height-scan columns)."""
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


class LocomotionPolicy(nn.Module):
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


# -----------------------
# Training and Logging
# -----------------------
def retrain_and_log(params, terrain_dirs, save_path):
    manifests = [build_manifest(terrain, data_dir) for terrain, data_dir in terrain_dirs.items()]
    manifest = pd.concat(manifests, axis=0, ignore_index=True)

    n_before = len(manifest)
    manifest = manifest[manifest["length"] >= MIN_TRAJ_LEN].reset_index(drop=True)
    n_dropped = n_before - len(manifest)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} degenerate trajectories with length < {MIN_TRAJ_LEN}")
    print(manifest.groupby("terrain")["length"].describe())

    traj_train, traj_test = train_test_split(
        manifest, test_size=0.1, stratify=manifest["terrain"], random_state=SEED
    )
    print(f"Train trajectories: {len(traj_train)} | Test trajectories: {len(traj_test)}")

    full_obs_cols = [f"obs_{i}" for i in range(1, 257)]

    X_trainval, y_trainval = load_trajectories(traj_train, full_obs_cols)
    X_test, y_test = load_trajectories(traj_test, full_obs_cols)

    terrain_list = sorted(terrain_dirs.keys())
    terrain_to_index = {name: idx for idx, name in enumerate(terrain_list)}
    
    os.makedirs("./scripts/UMBC/logs/models", exist_ok=True)
    with open("./scripts/UMBC/logs/models/terrain_mapping.json", "w") as f:
        json.dump(terrain_to_index, f)
    num_terrains = len(terrain_list)
    print(f"Terrain mapping: {terrain_to_index}")
    for df in (X_trainval, X_test):
        df["terrain_name"] = df["terrain_name"].map(terrain_to_index)

    model = LocomotionPolicy(num_terrains=num_terrains, embed_dim=8).to(device)

    X_train_t = torch.tensor(X_trainval.astype(np.float32).values).to(device)
    y_train_t = torch.tensor(y_trainval.astype(np.float32).values).to(device)
    X_test_t = torch.tensor(X_test.astype(np.float32).values).to(device)
    y_test_t = torch.tensor(y_test.astype(np.float32).values).to(device)

    assert X_train_t[:, 0].min() >= 0 and X_train_t[:, 0].max() < num_terrains, \
        "terrain_name column is misaligned — check load_trajectories column ordering"

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    # ---------------------------------------------------------
    # GROUPED BATCH SAMPLING
    # ---------------------------------------------------------
    train_labels = X_train_t[:, 0].cpu().numpy()
    all_batches = []
    for t_idx in range(num_terrains):
        idx = np.where(train_labels == t_idx)[0]
        sampler = SubsetRandomSampler(idx)
        all_batches += list(BatchSampler(sampler, batch_size=params["batch"], drop_last=True))
    np.random.shuffle(all_batches)

    train_loader = DataLoader(train_ds, batch_sampler=all_batches)
    test_loader = DataLoader(test_ds, batch_size=params["batch"], shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]) \
                if params["optimizer"] == "AdamW" else \
                optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.HuberLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'val_rmse': [], 'embeddings': []}
    history.update({f'val_{t}_loss': [] for t in terrain_list})

    terrain_weight = {t_idx: 1.0 for t_idx in range(num_terrains)}
    max_weight = 5.0

    best_val_loss = float('inf')
    best_state_dict = None
    patience = 8
    patience_counter = 0

    start_time = time.time()
    print(f"Starting training for up to {params['epoch']} epochs (early stopping patience={patience})...")

    for epoch in range(params['epoch']):
        model.train()
        train_run_loss = 0.0
        train_run_mae = 0.0

        for inputs, targets in train_loader:
            label_input = inputs[:, 0].long()
            proprio_input = inputs[:, 1:70]
            heightmap_input = inputs[:, 70:257]

            has_height = (label_input != terrain_to_index.get('flat', -1)).float().unsqueeze(1)
            noise = torch.randn_like(heightmap_input) * 0.02
            noisy_heights = heightmap_input + (noise * has_height)

            optimizer.zero_grad()
            outputs = model(proprio_input, noisy_heights, label_input)

            loss = criterion(outputs, targets)
            mae = F.l1_loss(outputs, targets)
            loss = loss * terrain_weight[label_input[0].item()]

            loss.backward()
            optimizer.step()

            train_run_loss += loss.item() * inputs.size(0)
            train_run_mae += mae.item() * inputs.size(0)

        # --- VALIDATION PHASE ---
        model.eval()
        val_run_loss = 0.0
        val_sq_err_sum = 0.0
        val_abs_err_sum = 0.0
        n_val_elems = 0
        per_terrain_loss = {t_idx: 0.0 for t_idx in range(num_terrains)}
        per_terrain_count = {t_idx: 0 for t_idx in range(num_terrains)}

        with torch.no_grad():
            for inputs, targets in test_loader:
                label_input = inputs[:, 0].long()
                proprio_input = inputs[:, 1:70]
                heightmap_input = inputs[:, 70:257]

                outputs = model(proprio_input, heightmap_input, label_input)

                batch_loss = F.huber_loss(outputs, targets, reduction='none').mean(dim=1)
                val_run_loss += batch_loss.sum().item()

                sq_err = (outputs - targets) ** 2
                abs_err = (outputs - targets).abs()
                val_sq_err_sum += sq_err.sum().item()
                val_abs_err_sum += abs_err.sum().item()
                n_val_elems += targets.numel()

                for t_idx in range(num_terrains):
                    mask = (label_input == t_idx)
                    per_terrain_loss[t_idx] += batch_loss[mask].sum().item()
                    per_terrain_count[t_idx] += mask.sum().item()

        avg_train_loss = train_run_loss / len(train_ds)
        avg_val_loss = val_run_loss / len(test_ds)
        val_mae = val_abs_err_sum / n_val_elems
        val_rmse = (val_sq_err_sum / n_val_elems) ** 0.5
        avg_terrain_loss = {
            t_idx: (per_terrain_loss[t_idx] / per_terrain_count[t_idx] if per_terrain_count[t_idx] > 0 else 0.0)
            for t_idx in range(num_terrains)
        }

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mae'].append(train_run_mae / len(train_ds))
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['embeddings'].append(model.terrain_embed.weight.detach().cpu().numpy().copy())
        for t_name, t_idx in terrain_to_index.items():
            history[f'val_{t_name}_loss'].append(avg_terrain_loss[t_idx])

        easiest_loss = min(v for v in avg_terrain_loss.values() if v > 0) if any(avg_terrain_loss.values()) else 1.0
        for t_idx, t_loss in avg_terrain_loss.items():
            if easiest_loss > 0 and t_loss > 0:
                target_w = max(1.0, min(max_weight, t_loss / easiest_loss))
                terrain_weight[t_idx] = 0.5 * terrain_weight[t_idx] + 0.5 * target_w

        scheduler.step(avg_val_loss)

        diag = " | ".join(f"{t_name}: {avg_terrain_loss[t_idx]:.6f}" for t_name, t_idx in terrain_to_index.items())
        print(f"Epoch {epoch+1}/{params['epoch']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} "
              f"| Val MAE: {val_mae:.6f} | Val RMSE: {val_rmse:.6f}")
        print(f"   >> [DIAGNOSTIC] {diag}")

        # --- early stopping ---
        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs). "
                      f"Best val loss: {best_val_loss:.6f}")
                break

    # restore best checkpoint before final evaluation/save
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    total_time = time.time() - start_time

    # ---- final per-motor / per-terrain action MAE & RMSE on held-out trajectories ----
    model.eval()
    with torch.no_grad():
        label_input = X_test_t[:, 0].long()
        proprio_input = X_test_t[:, 1:70]
        heightmap_input = X_test_t[:, 70:257]
        preds = model(proprio_input, heightmap_input, label_input)

        abs_err = (preds - y_test_t).abs()
        sq_err = (preds - y_test_t) ** 2

        per_motor_mae = abs_err.mean(dim=0).cpu().numpy()
        per_motor_rmse = torch.sqrt(sq_err.mean(dim=0)).cpu().numpy()
        overall_mae = abs_err.mean().item()
        overall_rmse = torch.sqrt(sq_err.mean()).item()

        print("=" * 60)
        print(f"FINAL TEST SET (best checkpoint) — Overall MAE: {overall_mae:.6f} | Overall RMSE: {overall_rmse:.6f}")
        motor_cols = y_test.columns.tolist()
        for col, mae_v, rmse_v in zip(motor_cols, per_motor_mae, per_motor_rmse):
            print(f"   {col:10s}  MAE: {mae_v:.6f}   RMSE: {rmse_v:.6f}")

        for t_name, t_idx in terrain_to_index.items():
            mask = (label_input == t_idx)
            if mask.sum() == 0:
                continue
            t_mae = abs_err[mask].mean().item()
            t_rmse = torch.sqrt(sq_err[mask].mean()).item()
            print(f"   [{t_name}]  MAE: {t_mae:.6f}   RMSE: {t_rmse:.6f}   (n={mask.sum().item()})")
        print("=" * 60)

    # --- SAVE MODEL WEIGHTS ---
    model_path = f"./scripts/UMBC/logs/models/{save_path}.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    # --- SAVE TRAINING HISTORY (PICKLE) ---
    history_path = f"./scripts/UMBC/logs/metrics/{save_path}_history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Saved training history to {history_path}")

    print("training_time_sec: ", total_time)
    return history['val_loss'][-1], overall_mae, overall_rmse


if __name__ == '__main__':
    best_params = {
        'lr': 0.0033876383257504254,
        'optimizer': 'AdamW',
        'batch': 128,
        'weight_decay': 0.0026553223463906626,
        'epoch': 60,
        'hidden_layer': 512
    }

    terrain_dirs = {
        'flat': './scripts/UMBC/dataset/flat',
        'stairway': './scripts/UMBC/dataset/stairway',
    }
    save_path = "adaptive_weight"

    final_val_loss, final_mae, final_rmse = retrain_and_log(best_params, terrain_dirs, save_path)
    print(f"Final Validation Loss: {final_val_loss:.6f} | Final MAE: {final_mae:.6f} | Final RMSE: {final_rmse:.6f}")
