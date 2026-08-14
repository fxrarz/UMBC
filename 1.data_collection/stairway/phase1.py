# ./isaaclab.sh -p scripts/UMBC/1.data_collection/stairway/phase1.py

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Phase1: Data collection of H1 stairway terrain expert."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import carb
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.nextgen.velocity.config.h1.stairway_env_cfg import H1StairwayEnvCfg_PLAY

TASK = "Isaac-Stairway-H1-Play-v0"
RL_LIBRARY = "rsl_rl"

# where to save trajectory files + the manifest/episode-length summary
OUT_DIR = "scripts/UMBC/dataset/stairway"


# ---------------------------------------------------------
# Helper function to save files in parallel
# ---------------------------------------------------------
def save_trajectory_worker(args):
    out_dir, env_id, traj_idx, obs_data, action_data = args
    if len(obs_data) == 0:
        return None
        
    obs_cols = [f"obs_{i}" for i in range(1, obs_data.shape[1] + 1)]
    action_cols = [f"motor_{i}" for i in range(1, action_data.shape[1] + 1)]
    
    obs_df = pd.DataFrame(obs_data, columns=obs_cols)
    action_df = pd.DataFrame(action_data, columns=action_cols)
    
    obs_df.insert(0, "step", range(len(obs_df)))
    action_df.insert(0, "step", range(len(action_df)))

    tag = f"env{env_id:03d}_traj{traj_idx:03d}"  # Updated to 3 digits for 500 envs
    obs_path = f"{out_dir}/obs_{tag}.csv"
    action_path = f"{out_dir}/action_{tag}.csv"
    
    obs_df.to_csv(obs_path, index=False)
    action_df.to_csv(action_path, index=False)
    
    return {
        "env_id": env_id, "traj_idx": traj_idx, "length": len(obs_df),
        "obs_file": os.path.basename(obs_path), "action_file": os.path.basename(action_path),
    }


class H1StairwayDemo:
    def __init__(self):
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        # load the trained jit policy
        checkpoint = "scripts/UMBC/logs/models/drl_stairway.pt"
        # create envionrment
        env_cfg = H1StairwayEnvCfg_PLAY()
        
        # SCALED UP TO 500 ENVS
        env_cfg.scene.num_envs = 500
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.env_spacing = 0.01
        env_cfg.episode_length_s = 5

        # ---------------------------------------------------------
        # LOCAL OVERRIDE: Modify Sub-Terrain Border Width
        # ---------------------------------------------------------
        DESIRED_SUB_TERRAIN_BORDER_WIDTH = 0.2  # Set your desired border width here
        
        # Safely overwrite the border width for the specific stair terrains
        sub_terrains = env_cfg.scene.terrain.terrain_generator.sub_terrains
        for terrain_key in ["pyramid_stairs", "pyramid_stairs_inv"]:
            if terrain_key in sub_terrains:
                sub_terrains[terrain_key].border_width = DESIRED_SUB_TERRAIN_BORDER_WIDTH
        
        # ---------------------------------------------------------
        # LOCAL OVERRIDE: Randomize Spawn Locations on Reset
        # ---------------------------------------------------------
        # Re-enable pose randomization which is normally disabled in PLAY mode.
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base"):
            env_cfg.events.reset_base.params["pose_range"] = {
                "x": (-0.5, 0.5),     # Randomize X offset by ±0.5 meters
                "y": (-0.5, 0.5),     # Randomize Y offset by ±0.5 meters
                "yaw": (-3.14, 3.14)  # Randomize starting rotation (full 360 degrees)
            }

        # 1. Set forward speed (X) to your desired range
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        # 2. Force lateral velocity (Y) to zero
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # 3. Force heading/yaw rate to zero to prevent turning
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # wrap around environment for rsl-rl
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        # load previously trained model
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
        # obtain the trained policy for inference
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 4, device=self.device)
        self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)
    def create_camera(self):
        """Creates a camera to be used for third-person view."""
        stage = omni.usd.get_context().get_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        # Create camera
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
        """Sets up interface for keyboard input and registers the desired keys for control."""
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
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
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
            # a valid robot was selected, update the camera to go into third-person view
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a H1 robot")
        # Reset commands for previously selected robot if a new one is selected
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")

    def _update_camera(self):
        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]  # - env.scene.env_origins
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]
        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos
        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)


def main():
    demo_h1 = H1StairwayDemo()
    obs, _ = demo_h1.env.reset()
    num_envs = demo_h1.env.unwrapped.num_envs

    cfg = demo_h1.env.unwrapped.cfg
    print("=" * 60)
    print(f"num_envs           : {num_envs}")
    print(f"episode_length_s   : {cfg.episode_length_s}")
    print(f"sim.dt             : {cfg.sim.dt}")
    print(f"decimation         : {cfg.decimation}")
    print(f"max_episode_length (steps): {demo_h1.env.unwrapped.max_episode_length}")
    print("=" * 60)

    # ---------------------------------------------------------
    # GOAL: Collect exactly 2000 successful full-length trajectories
    # With 500 envs, this will finish in just 4 successful iterations!
    # ---------------------------------------------------------
    TARGET_SUCCESSFUL_TRAJECTORIES = 4000

    completed_trajectories = []
    traj_counter = {i: 0 for i in range(num_envs)}
    step_counters = torch.zeros(num_envs, dtype=torch.long)
    episode_lengths = []
    episode_was_timeout = []

    # Use lists to store numpy arrays temporarily
    current_obs = [[] for _ in range(num_envs)]
    current_actions = [[] for _ in range(num_envs)]
    
    successful_trajs_collected = 0
    t = 0
    
    start_time = time.time()
    
    # Run loop until we gather enough successful trajectories
    while successful_trajs_collected < TARGET_SUCCESSFUL_TRAJECTORIES:
        if t % 500 == 0:
            print(f"Step: {t} | Successful Trajectories: {successful_trajs_collected}/{TARGET_SUCCESSFUL_TRAJECTORIES}")
            
        demo_h1.update_selected_object()
        
        with torch.inference_mode():
            action = demo_h1.policy(obs)

            # MOVE BATCH TO CPU ONCE PER STEP
            obs_cpu = obs.cpu().numpy()
            action_cpu = action.cpu().numpy()

            for env_id in range(num_envs):
                current_obs[env_id].append(obs_cpu[env_id])
                current_actions[env_id].append(action_cpu[env_id])
                
            step_counters += 1
            obs, rewards, dones, extras = demo_h1.env.step(action)

            done_env_ids = torch.nonzero(dones).flatten().tolist()
            if len(done_env_ids) > 0:
                time_outs = extras.get("time_outs", None) if isinstance(extras, dict) else None
                for env_id in done_env_ids:
                    ep_length = step_counters[env_id].item()
                    is_timeout = bool(time_outs[env_id].item()) if time_outs is not None else False
                    
                    episode_lengths.append(ep_length)
                    episode_was_timeout.append(is_timeout)
                    
                    # ONLY save if it ended via timeout (survived maximum time)
                    if is_timeout and len(current_obs[env_id]) > 0:
                        completed_trajectories.append((
                            env_id, 
                            traj_counter[env_id],
                            np.array(current_obs[env_id]),
                            np.array(current_actions[env_id])
                        ))
                        traj_counter[env_id] += 1
                        successful_trajs_collected += 1
                    
                    # Clear the buffers (drops fallen trajectories automatically)
                    current_obs[env_id] = []
                    current_actions[env_id] = []
                    step_counters[env_id] = 0
                    
                    # Break out early if we hit the target during this batch
                    if successful_trajs_collected >= TARGET_SUCCESSFUL_TRAJECTORIES:
                        break
            
            # --- Stairway specific custom command injection ---
            obs[:, 9:13] = demo_h1.commands
            
        t += 1
        if successful_trajs_collected >= TARGET_SUCCESSFUL_TRAJECTORIES:
            break

    print(f"Simulation took {time.time() - start_time:.2f} seconds")

    # ---------------------------------------------------------
    # PRINT EXACT REQUESTED STATS
    # ---------------------------------------------------------
    if len(episode_lengths) > 0:
        ep = pd.Series(episode_lengths)
        n_timeout = sum(episode_was_timeout)
        n_fell = len(episode_was_timeout) - n_timeout
        
        print("=" * 60)
        print("Completed episode length stats (steps):")
        print(ep.describe())
        print("Percentiles:")
        print(ep.quantile([0.10, 0.25, 0.50, 0.75, 0.90]))
        print(f"Ended via timeout: {n_timeout} | Ended via early termination (e.g. fell): {n_fell}")
        print("=" * 60)

    print("Started processing the data")

    os.makedirs(OUT_DIR, exist_ok=True)
    save_start_time = time.time()
    
    save_args = [
        (OUT_DIR, env_id, traj_idx, obs_data, act_data) 
        for env_id, traj_idx, obs_data, act_data in completed_trajectories
    ]

    manifest_rows = []
    with ProcessPoolExecutor() as executor:
        for result in executor.map(save_trajectory_worker, save_args):
            if result is not None:
                manifest_rows.append(result)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(f"{OUT_DIR}/manifest.csv", index=False)
    
    print(f"Saved {len(manifest_rows)} successful full-length trajectories to {OUT_DIR}")
    print(f"File saving took {time.time() - save_start_time:.2f} seconds")


if __name__ == "__main__":
    main()
    simulation_app.close()
