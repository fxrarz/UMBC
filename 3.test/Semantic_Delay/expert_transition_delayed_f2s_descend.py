# python3 scripts/UMBC/4.result/7.expert_transition_delayed_f2s_descend.py --delay 25

# only have 'pyramid_stairs' in sub-terrain. /home/whitebot/Workspace/Research/IsaacLab/source/isaaclab/isaaclab/terrains/config/stairway.py

import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="This script demonstrates an interactive demo with the H1 flat terrain environment."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--delay", type=int, default=100,
                    help="Set delay")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import torch.nn as nn
import carb
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.nextgen.velocity.config.h1.stairway_env_cfg import H1StairwayEnvCfg_PLAY
from rsl_rl.modules import ActorCritic
import pickle
import json

TASK_Stairway = "Isaac-Stairway-H1-Play-v0"
TASK_Flat = "Isaac-Velocity-Flat-H1-v0"
RL_LIBRARY = "rsl_rl"

delay = args_cli.delay

class H1FlatDemo:
    def __init__(self):
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK_Stairway, args_cli)
#        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
        env_cfg = H1StairwayEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1

        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.env_spacing = 0.01
        env_cfg.episode_length_s = 5
        
        env_cfg.scene.terrain.terrain_generator.num_rows = 50
        env_cfg.scene.terrain.terrain_generator.num_cols = 1
        
        env_cfg.terminations.root_height_below_minimum = None

        # ---------------------------------------------------------
        # LOCAL OVERRIDE: Modify Sub-Terrain Border Width
        # ---------------------------------------------------------
        DESIRED_SUB_TERRAIN_BORDER_WIDTH = 1.5  # Set your desired border width here
        
        # Safely overwrite the border width for the specific stair terrains
        sub_terrains = env_cfg.scene.terrain.terrain_generator.sub_terrains
        for terrain_key in ["pyramid_stairs", "pyramid_stairs_inv"]:
            if terrain_key in sub_terrains:
                sub_terrains[terrain_key].border_width = DESIRED_SUB_TERRAIN_BORDER_WIDTH

        # ---------------------------------------------------------
        # LOCAL OVERRIDE: Fixed Spawn Location on Reset
        # ---------------------------------------------------------
        # 1. Disable all randomization by setting ranges to exactly 0.0
#        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base"):
#            env_cfg.events.reset_base.params["pose_range"] = {
#                "x": (0.0, 0.0),     
#                "y": (0.0, 0.0),     
#                "yaw": (0.0, 0.0),
#                "roll": (0.0, 0.0),
#                "pitch": (0.0, 0.0)
#            }
        
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base"):
            env_cfg.events.reset_base.params["pose_range"] = {
                "x": (-0.5, 0.5),     # Randomize X offset by ±0.5 meters
                "y": (-0.5, 0.5),     # Randomize Y offset by ±0.5 meters
                "yaw": (-3.14, 3.14)  # Randomize starting rotation (full 360 degrees)
            }            
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.05) 
        env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0) # Identity quaternion (facing straight ahead)
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

            
        # 2. Hardcode the exact absolute starting coordinates (X, Y, Z) and rotation (W, X, Y, Z)
        # Note: Ensure the Z height is appropriate for the H1 robot so it doesn't clip into the ground
#        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.05) 
#        env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0) # Identity quaternion (facing straight ahead)

        # 1. Set forward speed (X) to your desired range
#        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        # 2. Force lateral velocity (Y) to zero
#        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # 3. Force heading/yaw rate to zero to prevent turning
#        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using {device} device")

        weights_path = "scripts/UMBC/logs/models/drl_stairway.pt"
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(weights_path)
        self.policy_stair = ppo_runner.get_inference_policy(device=self.device)

        weights_path = "scripts/UMBC/logs/models/drl_flat.pt"
        ppo_runner_flat = ActorCritic(
            num_actor_obs=69,
            num_critic_obs=69,
            num_actions=19,
            actor_hidden_dims=[128, 128, 128],
            critic_hidden_dims=[128, 128, 128],
            activation="elu",
        ).to(device)
        
        checkpoint = torch.load(
            weights_path,
            map_location=device,
            weights_only=False,
        )
        ppo_runner_flat.load_state_dict(checkpoint["model_state_dict"])
        self.policy_flat = ppo_runner_flat.eval()

        self.create_camera()
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)
    def create_camera(self):
        stage = omni.usd.get_context().get_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)
    def set_up_keyboard(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        T = 1
        R = 0.5
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([T, 0.0, 0.0, -R], device=self.device),
            "RIGHT": torch.tensor([T, 0.0, 0.0, R], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
        }
    def _on_keyboard_event(self, event):
         if event.type == carb.input.KeyboardEventType.KEY_PRESS:
             if event.input.name in self._key_to_control:
                if self._selected_id:
                    self.commands[self._selected_id] = self._key_to_control[event.input.name]
             elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
             elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
         elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if self._selected_id:
                self.commands[self._selected_id] = self._key_to_control["ZEROS"]
    def update_selected_object(self):
        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a H1 robot")
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")
    def _update_camera(self):
        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]
        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos
        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)

def main():
    import matplotlib.pyplot as plt
    import numpy as np
    
    # --- LOGGING MASTER LISTS ---
    rewards = []
    states = []
    actions = []
    terrains = []
    reward_split = {}  # built dynamically from info['log'] on first episode end

    # --- LOGGING EPISODE LISTS ---
    reward_list = []
    state_list = []
    action_list = []
    terrain_list = []
    
    fall_count = 0
    fall_flags = []  # per-episode: True if that episode ended in a fall


    demo_h1 = H1FlatDemo()
    obs, _ = demo_h1.env.reset()
#    print("root height right after reset:", demo_h1.env.unwrapped.scene["robot"].data.root_pos_w[:, 2].item())

    number_envs = 1

    # --- Live Visualization Setup ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 5))
    h_grid_init = np.zeros((11, 17))
    img_plot = ax.imshow(h_grid_init, vmin=-0.5, vmax=1.0, cmap='viridis', origin='lower')
    ax.set_title("Live Heightmap (11x17)")
    ax.set_xlabel("Lateral (17 cols)")
    ax.set_ylabel("Forward (11 rows)")
    plt.colorbar(img_plot, ax=ax, label="Height (m)")
    plt.show()

    step_counter = 0
    HEIGHT_TOLERANCE = 0.02 # 2 cm tolerance for local unevenness
    
    time_step = 0
    switch = True

    while simulation_app.is_running():
#        print(time_step)
#        time_step+=1

        demo_h1.update_selected_object()
        with torch.inference_mode():
            height = obs[:, 69:]
            
            # Reshape the 1D heightmap into the 11x17 grid
            h_grid = height[0].view(11, 17)
            
            # --- Live Plot Update (every 3 steps to avoid lag) ---
            if step_counter % 3 == 0:
                h_grid_np = h_grid.cpu().numpy()
                img_plot.set_data(h_grid_np)
                img_plot.set_clim(vmin=h_grid_np.min(), vmax=h_grid_np.max() + 0.1)
                fig.canvas.draw()
                fig.canvas.flush_events()
            
            # --- Dynamic Terrain Logic (Baseline-Free) ---
            front_slice = h_grid[:, 10:]  
            
            # Find the "rest of the pixels" (the majority height in this exact frame)
            local_median = torch.median(front_slice)
            
            # Check which points deviate from this local majority by more than 2cm
            changed_points = torch.abs(front_slice - local_median) > HEIGHT_TOLERANCE
            
            # Calculate the percentage of deviating points
            percent_changed = changed_points.float().mean().item()
            
            # Memoryless switch: purely based on the current frame's 5% variance rule
            if percent_changed > 0.05:
                time_step+=1
                if time_step > delay:
                    terrain_id = 1  # stairway
                    switch = False
                    print("switch disabled")                    
            else:
                time_step = 0
                if switch == False:
                    terrain_id = 1  # flat
                else:
                    terrain_id = 0
                # Zero out the heightmap input when assuming flat ground
                height = torch.full((number_envs, 187), 0, device='cuda:0', dtype=torch.float32)
                
            
            terrain_str = "Stairway" if terrain_id == 1 else "Flat"
            print(f"[{terrain_str}] Variance from median: {percent_changed*100:.2f}% {time_step}")
            # -----------------------------
            
            if terrain_id == 0:
                action = demo_h1.policy_flat.act_inference(obs[:, :69])
            else:
                action = demo_h1.policy_stair(obs)
            obs, reward, done, info = demo_h1.env.step(action)
            
            if done[0] != 0:  # reset
#                print("root height right after reset:", demo_h1.env.unwrapped.scene["robot"].data.root_pos_w[:, 2].item())
                obs, _ = demo_h1.env.reset()
                switch = True
                if 'log' in info:
                    for k, v in info['log'].items():
                        reward_split.setdefault(k, []).append(v)
                
                fell = float(info['log'].get('Episode_Termination/base_contact', 0.0)) > 0
                fall_flags.append(fell)
                if fell:
                    fall_count += 1
                
                # Append episode data to master lists
                rewards.append(reward_list)
                states.append(state_list)
                actions.append(action_list)
                terrains.append(terrain_list)
                
                # Clear episode buffers
                reward_list = []
                state_list = []
                action_list = []
                terrain_list = []
                if len(rewards) == 20:
                    break
            else:
                # Log current step data (converted to CPU NumPy arrays for easy plotting)
                reward_list.append(reward[0].item())
                state_list.append(obs[0].cpu().numpy())
                action_list.append(action[0].cpu().numpy())
                terrain_list.append(terrain_id)
                
            step_counter += 1

    # Package all logged data into the results dictionary
    results = {
        'rewards': rewards, 
        'individual_rewards': reward_split,
        'states': states,
        'actions': actions,
        'terrains': terrains
    }
    
    import os, pickle
    os.makedirs("scripts/UMBC/logs/metrics", exist_ok=True)
    with open(f"scripts/UMBC/logs/metrics/expert_transition_delay_f2s_descend_{delay}.pkl", "wb") as f:
        pickle.dump(results, f)

    print('===============')
    print(f'UMBC Delayed Flat2Stair Descend - {delay} timestep delay')
    print('===============')
    print('Number of episodes: ', len(rewards))
    print('Fall count: ', fall_count)
    print('Fall rate: ', fall_count / len(rewards))
    print('Length of episode: ', [len(r) for r in rewards])
    print('Average length of episode: ', sum(len(r) for r in rewards) / len(rewards))
    print('Per episode reward: ', [sum(r) for r in rewards])
    print('Average reward: ', sum(sum(r) for r in rewards) / len(rewards))
    print('Total reward: ', sum(sum(r) for r in rewards), '\n\n')


if __name__ == "__main__":
    main()
    simulation_app.close()

