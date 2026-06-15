'''
python3 UMBC/experimental/train.py
'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mlflow
import time
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# -----------------------
# Global config
# -----------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

mlflow.set_tracking_uri("file:./UMBC/logs/mlruns")
mlflow.set_experiment("Behaviour Cloning")

def get_data(terrain, observation, action, batch_size=32, input_dim=257): # zero padding
    observation_df = pd.DataFrame({'terrain_name':[]})
    action_df = pd.DataFrame()
    for terrain in zip(terrain, observation, action):
      obs_df = pd.read_csv(terrain[1]) # ex: obs
      act_df = pd.read_csv(terrain[2]) # ex: action
      index_to_drop = [i for i in range(act_df.shape[0], obs_df.shape[0])]
      obs_df.drop(labels=index_to_drop, axis='index', inplace=True)
      obs_df['terrain_name'] = terrain[0]
      observation_df = pd.concat([observation_df, obs_df], axis=0)
      action_df = pd.concat([action_df, act_df], axis=0)
    observation_df.fillna(0, inplace=True)
    observation_df.reset_index(drop=True, inplace=True)
    action_df.reset_index(drop=True, inplace=True)
    return observation_df, action_df

class LocomotionPolicy(nn.Module):
    def __init__(self, proprio_dim=69, heightmap_dim=187, output_dim=19):
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
        self.terrain_embed = nn.Embedding(2, 8)        
        self.final_head = nn.Sequential(
            nn.Linear(128 + 64 + 8, 128),
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
def retrain_and_log(study_name, run_name, params, dataset, save_path):
    observation_list = dataset['observation_list']
    action_list = dataset['action_list']
    X, y = get_data(terrain_list, observation_list, action_list)
    #y = y[X.terrain_name == 'stairway']
    #X = X[X.terrain_name == 'stairway']
    terrain_to_index = {name: idx for idx, name in enumerate(terrain_list)}
    X["terrain_name"] = X["terrain_name"].map(terrain_to_index)
    print(f"Terrain mapping: {terrain_to_index}")
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.1, stratify=X['terrain_name'], random_state=SEED
    )
    
    model = LocomotionPolicy().to(device)
    
    # Prep Tensors
    X_train_t = torch.tensor(X_trainval.astype(np.float32).values).to(device)
    y_train_t = torch.tensor(y_trainval.astype(np.float32).values).to(device)
    X_test_t = torch.tensor(X_test.astype(np.float32).values).to(device)
    y_test_t = torch.tensor(y_test.astype(np.float32).values).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    # ---------------------------------------------------------
    # MODIFIED: GROUPED BATCH SAMPLING
    # ---------------------------------------------------------
    # 1. Identify indices for each terrain type
    # label_input is at index 0 based on your training loop slicing
    train_labels = X_train_t[:, 0].cpu().numpy()
    flat_idx = np.where(train_labels == 0)[0]
    stair_idx = np.where(train_labels == 1)[0]

    # 2. Create Random Samplers for each group
    from torch.utils.data import BatchSampler, SubsetRandomSampler
    flat_sampler = SubsetRandomSampler(flat_idx)
    stair_sampler = SubsetRandomSampler(stair_idx)

    # 3. Create BatchSamplers (Fixed-terrain batches)
    flat_batches = list(BatchSampler(flat_sampler, batch_size=params["batch"], drop_last=True))
    stair_batches = list(BatchSampler(stair_sampler, batch_size=params["batch"], drop_last=True))

    # 4. Combine and shuffle the batches themselves 
    # (The model sees a mix of batches, but each batch is internally pure)
    all_batches = flat_batches + stair_batches
    np.random.shuffle(all_batches)

    # 5. Initialize DataLoader with the custom batch_sampler
    train_loader = DataLoader(train_ds, batch_sampler=all_batches)
    # Validation stays standard to check overall performance
    test_loader = DataLoader(test_ds, batch_size=params["batch"], shuffle=False)
    # ---------------------------------------------------------

    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]) \
                if params["optimizer"] == "AdamW" else \
                optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, 
		mode='min', 
		factor=0.5,    # Reduce LR by half
		patience=5,    # Wait 5 epochs before reducing
		verbose=True   # Print a message when LR changes
	)
    criterion = nn.HuberLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'val_flat_loss': [], 'val_stair_loss': []}
    
    start_time = time.time()
    print(f"Starting training for {params['epoch']} epochs with Pure-Batching...")
    
    stair_weight = 1.0
    max_weight = 5.0
    
    # start time
    start_time = time.time()
    
    for epoch in range(params['epoch']):
        model.train()
        train_run_loss = 0.0
        train_run_mae = 0.0
        
        for inputs, targets in train_loader:
            # Split of observation
            label_input = inputs[:, 0].long()
            proprio_input = inputs[:, 1:70]
            heightmap_input = inputs[:, 70:257]

            # Selective Noise for Stairs
            stair_mask = (label_input == 1).float().unsqueeze(1)
            noise = torch.randn_like(heightmap_input) * 0.02
            noisy_heights = heightmap_input + (noise * stair_mask)

            optimizer.zero_grad()
            outputs = model(proprio_input, noisy_heights, label_input)
            
            loss = criterion(outputs, targets)
            mae = F.l1_loss(outputs, targets)
            
            if label_input[0].item() == 1:
                loss = loss * stair_weight
            
            loss.backward()
            optimizer.step()

            train_run_loss += loss.item() * inputs.size(0)
            train_run_mae += mae.item() * inputs.size(0)

        # --- VALIDATION PHASE ---

        model.eval()
        val_run_loss = 0.0
        val_run_mae = 0.0
        
        # Diagnostic trackers
        flat_loss_total = 0.0
        stair_loss_total = 0.0
        flat_count = 0
        stair_count = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                label_input = inputs[:, 0].long()
                proprio_input = inputs[:, 1:70]
                heightmap_input = inputs[:, 70:257]

                outputs = model(proprio_input, heightmap_input, label_input)
                
                # Calculate individual losses for the batch
                batch_loss = F.huber_loss(outputs, targets, reduction='none').mean(dim=1)
                val_run_loss += batch_loss.sum().item()
                
                # Calculate MAE (L1)
                batch_mae = F.l1_loss(outputs, targets, reduction='none').mean(dim=1)
                val_run_mae += batch_mae.sum().item()

                # Diagnostic: Split batch by label
                is_flat = (label_input == 0)
                is_stair = (label_input == 1)
                
                flat_loss_total += batch_loss[is_flat].sum().item()
                stair_loss_total += batch_loss[is_stair].sum().item()
                
                flat_count += is_flat.sum().item()
                stair_count += is_stair.sum().item()

        # Calculate Averages
        avg_val_loss = val_run_loss / len(test_ds)
        avg_flat_loss = flat_loss_total / flat_count if flat_count > 0 else 0
        avg_stair_loss = stair_loss_total / stair_count if stair_count > 0 else 0

        history['train_loss'].append(train_run_loss / len(train_ds))
        history['val_loss'].append(avg_val_loss)
        history['train_mae'].append(train_run_mae / len(train_ds))
        history['val_mae'].append(val_run_mae / len(test_ds))
        history['val_flat_loss'].append(avg_flat_loss)
        history['val_stair_loss'].append(avg_stair_loss)
        
        if avg_flat_loss > 0:
            # Calculate the ratio of how much harder stairs are than flat ground
            raw_ratio = avg_stair_loss / avg_flat_loss
            # We use a moving average or a dampened update so the weight doesn't jump too fast
            target_weight = max(1.0, min(max_weight, raw_ratio))
            
            # Dampening (optional but recommended): 50% old weight, 50% new ratio
            stair_weight = 0.5 * stair_weight + 0.5 * target_weight

        scheduler.step(avg_val_loss)
        
        
        print(f"Epoch {epoch+1}/{params['epoch']} | Total Val Loss: {avg_val_loss:.6f}")
        print(f"   >> [DIAGNOSTIC] Flat Loss: {avg_flat_loss:.6f} | Stair Loss: {avg_stair_loss:.6f}")
    
    # stop time    
    total_time = time.time() - start_time

    # save model
    model_path = f"UMBC/logs/policy/{save_path}.pth"
    #torch.save(model.state_dict(), model_path)

    epochs_range = range(1, params['epoch'] + 1)

    # --- Plotting ---
    # 1st Graph: Train Loss vs Val Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], 'r--', label='Val Loss')
    plt.title(f'Overall Loss - {run_name}')
    plt.xlabel('Epochs'); plt.ylabel('Huber Loss')
    plt.legend(); plt.grid(True)
    plot_path = f"UMBC/logs/imgs/{save_path}_overall_loss.png"
    plt.savefig(plot_path)
    plt.close()
    
    # 2nd Graph: Val Loss vs Flat Loss vs Stairway Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['val_loss'], 'k-', alpha=0.3, label='Total Val')
    plt.plot(epochs_range, history['val_flat_loss'], 'g--', label='Flat Ground')
    plt.plot(epochs_range, history['val_stair_loss'], 'r--', label='Stairway')
    plt.title(f'Terrain-Specific Loss - {run_name}')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plot_path = f"UMBC/logs/imgs/{save_path}_terrain_breakdown.png"
    plt.savefig(plot_path)
    plt.close()
    
    # 3rd Graph: Train MAE vs Val MAE
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['train_mae'], 'b-', label='Train MAE')
    plt.plot(epochs_range, history['val_mae'], 'orange', linestyle='--', label='Val MAE')
    plt.title(f'Precision (MAE) - {run_name}')
    plt.xlabel('Epochs'); plt.ylabel('MAE')
    plt.legend(); plt.grid(True)
    plot_path = f"UMBC/logs/imgs/{save_path}_mae_precision.png"
    plt.savefig(plot_path)
    plt.close()

    # --- MLflow Logging ---
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metric("final_val_loss", history['val_loss'][-1])
        mlflow.log_metric("final_stair_loss", avg_stair_loss)
        mlflow.log_metric("final_flat_loss", avg_flat_loss)
        mlflow.log_metric("training_time_sec", total_time)
        mlflow.log_artifact(plot_path)      
        mlflow.log_artifact(model_path)

        mlflow.set_tag("stage", "deploy")
        mlflow.set_tag("study_name", study_name)

    print("training_time_sec: ", total_time)
    return history['val_loss'][-1]
    
    
if __name__ == '__main__':
	# -----------------------
	# Execution
	# -----------------------
	best_params = {
		'lr': 0.0033876383257504254,
		'optimizer': 'AdamW',
		'batch': 128,
		'weight_decay': 0.0026553223463906626,
		'epoch': 15,
		'hidden_layer': 512
	}

	study_name = 'Semantic_ID'
	terrain_list = ['flat', 'stairway']

	run_name = 'Dev'
	observation_list = ['UMBC/logs/dataset/flat/obs_df.csv', 'UMBC/logs/dataset/stairway/obs_df.csv']
	action_list = ['UMBC/logs/dataset/flat/action_df.csv', 'UMBC/logs/dataset/stairway/action_df.csv']
	save_path = "UMBC_policy"

	dataset = {'observation_list': observation_list, 'action_list': action_list}
	final_val_loss = retrain_and_log(study_name, run_name, best_params, dataset, save_path)
	print(f"Final Validation Loss: {final_val_loss:.6f}")
