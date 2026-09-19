###
# Figure 5: Reward decomposition for the flat-ground reference policy and UMBC policy.
# Figure 6: Reward decomposition for the stairway reference policy and UMBC policy.
###

# instantaneous
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# 1. Configuration
base_path = "scripts/UMBC/logs/metrics/"
output_dir = "scripts/UMBC/logs/images/"

# Corrected grouping structure
groups = {
    "Flat": {
        "exp_key": "Flat Ground Reference Policy",
        "exp_file": "expert_flat_eval.pkl",
        "bc_key": "UMBC Policy",
        "bc_file": "umbc_flat_eval.pkl",
        "colors": ['#2ecc71', '#3498db'] # Expert Green, BC Blue
    },
    "Stairway": {
        "exp_key": "Stairway Reference Policy",
        "exp_file": "expert_stairway_eval.pkl",
        "bc_key": "UMBC Policy",
        "bc_file": "umbc_stairway_eval.pkl",
        "colors": ['#27ae60', '#2980b9'] # Darker shades for stairs
    }
}

def get_clean_metrics(data):
    reward_data = data['individual_rewards']
    metrics = {}
    for key, values in reward_data.items():
        if 'Episode_Reward/' in key:
            clean_name = key.replace('Episode_Reward/', '')
            # Handle list of tensors or single values
            avg_val = np.mean([v.item() if torch.is_tensor(v) else v for v in values])
            metrics[clean_name] = avg_val
    return metrics

layout = [
    ['A', 'A', 'A', 'A'],
    ['B', 'C', 'D', 'E'],
    ['F', 'G', 'H', 'I'],
    ['J', 'K', 'L', 'M'],
]

# 2. Process each Terrain Group
for terrain, config in groups.items():
    # FIX: Access keys using the names defined in the dictionary (exp_file, bc_file)
    path_exp = os.path.join(base_path, config["exp_file"])
    path_bc = os.path.join(base_path, config["bc_file"])
    data_exp = get_clean_metrics(pickle.load(open(path_exp, "rb")))
    data_bc = get_clean_metrics(pickle.load(open(path_bc, "rb")))
    # Ensure both have the same keys (sorted by Expert values for readability)
    all_keys = sorted(data_exp.keys(), key=lambda k: data_exp[k])
    exp_vals = [data_exp[k] for k in all_keys[-2:]]
    bc_vals = [data_bc.get(k, 0) for k in all_keys[-2:]] 
    # layout
    fig, axes = plt.subplot_mosaic(layout, figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    # --- Plotting 1st row Chart ---
    ax1 = axes['A']
    y_pos = np.arange(len(all_keys[-2:]))
    height = 0.35  
    # Draw Bars using the custom keys for labels
    ax1.barh(y_pos + height/2, exp_vals, height, label=config["exp_key"], color=config["colors"][0], alpha=0.8)
    ax1.barh(y_pos - height/2, bc_vals, height, label=config["bc_key"], color=config["colors"][1], alpha=0.8)
    # Vertical line at zero
    ax1.axvline(0, color='black', linewidth=1.2)
    # Labels and Titles
    # ax1.set_yticks(y_pos, all_keys[-2:])
    ax1.set_yticks([])
    ax1.set_title('Flat Terrain Reward Split', fontsize=16, fontweight='bold')
#    ax1.set_title('Stairway Terrain Reward Split', fontsize=16, fontweight='bold')
    #ax1.set_xlabel('Mean Reward Value')
    # ax1.legend(loc="lower left")
    ax1.legend(bbox_to_anchor=(0.5, 1.55), loc='upper center', ncol=3)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    # Add numeric labels to BC bars
    for i, val in enumerate(zip(bc_vals, exp_vals)):
        ax1.text(val[0], i - height/2, f' {val[0]:.3f}', va='center', fontsize=10, color='black', fontweight='bold')
        ax1.text(val[1], i + height/2, f' {val[1]:.3f}', va='center', fontsize=10, color='black', fontweight='bold')
    ax1.text(0.4, 0, f' {all_keys[-2]}', va='center', fontsize=10, color='black', fontweight='bold')
    ax1.text(0.4, 1, f' {all_keys[-1]}', va='center', fontsize=10, color='black', fontweight='bold')
    # --- Plotting 2nd row Chart ---
    remaining_keys = all_keys[:-2]
    subplot_keys = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    for i, key in enumerate(subplot_keys):
        if i < len(remaining_keys):
            ax = axes[key]
            metric = remaining_keys[i]
            ax.bar(0, data_exp[metric], width=0.35, color=config["colors"][0], label='Expert', alpha=0.8)
            ax.bar(0.4, data_bc.get(metric, 0), width=0.35, color=config["colors"][1], label='BC', alpha=0.8)
#            if key == 'J' and terrain == 'Stairway':
#                ax.text(0, data_exp[metric] - 0.0002, f'{data_exp[metric]:.3f}', ha='center', va='top', fontsize=10, fontweight='bold')
#                ax.text(0.4, data_bc.get(metric, 0) + 0.0007, f'{data_bc.get(metric, 0):.3f}', ha='center', va='top', fontsize=10, fontweight='bold')
            if key == 'K' and terrain == 'Stairway':
                ax.text(0, data_exp[metric], f'{data_exp[metric]:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax.text(0.4, data_bc.get(metric, 0), f'{data_bc.get(metric, 0):.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            elif key == 'M' and terrain == 'Stairway':
                ax.text(0, -0.0025 + data_exp[metric], f'{data_exp[metric]:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax.text(0.4, -0.0025 + data_bc.get(metric, 0), f'{data_bc.get(metric, 0):.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            elif key == 'K' and terrain == 'Flat':
                ax.text(0, data_exp[metric], f'{data_exp[metric]:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax.text(0.4, data_bc.get(metric, 0), f'{data_bc.get(metric, 0):.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            elif key == 'L' and terrain == 'Flat':
                ax.text(0, data_exp[metric], f'{data_exp[metric]:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax.text(0.4, data_bc.get(metric, 0), f'{data_bc.get(metric, 0):.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            elif key == 'M' and terrain == 'Flat':
                ax.text(0, -0.010 + data_exp[metric], f'{data_exp[metric]:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax.text(0.4, -0.010 + data_bc.get(metric, 0), f'{data_bc.get(metric, 0):.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')                
            else:
                ax.text(0, data_exp[metric], f'{data_exp[metric]:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax.text(0.4, data_bc.get(metric, 0), f'{data_bc.get(metric, 0):.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.set_title(metric, fontsize=10, fontweight='bold')
            ax.set_xticks([]) # No ticks needed for a single category
            ax.set_xlim(-0.5, 0.9)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
    # plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.47, hspace=0.35)
    # Save the comparison plot
    save_path = os.path.join(output_dir, f"8.{terrain.lower()}_reward_split.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
