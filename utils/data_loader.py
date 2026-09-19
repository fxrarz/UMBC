import os
import glob
import re
import pandas as pd

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

