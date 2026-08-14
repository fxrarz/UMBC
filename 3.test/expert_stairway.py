# python3 scripts/UMBC/3.test/expert_stairway.py

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
import pickle

TASK = "Isaac-Stairway-H1-Play-v0"
RL_LIBRARY = "rsl_rl"

class H1FlatDemo:
    def __init__(self):
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
        env_cfg = H1StairwayEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1

        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.env_spacing = 0.01
        env_cfg.episode_length_s = 5

        env_cfg.terminations.root_height_below_minimum = None

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

        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        checkpoint = "logs/rsl_rl/h1_stairway/2025-12-12_02-09-27/model_2999.pt"
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
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
    rewards = []
    reward_list = []
    reward_split = {'Episode_Reward/track_lin_vel_xy_exp':[], 'Episode_Reward/track_ang_vel_z_exp': [], 'Episode_Reward/ang_vel_xy_l2': [], 'Episode_Reward/dof_torques_l2': [], 'Episode_Reward/dof_acc_l2': [], 'Episode_Reward/action_rate_l2': [], 'Episode_Reward/feet_air_time': [], 'Episode_Reward/flat_orientation_l2': [], 'Episode_Reward/dof_pos_limits': [], 'Episode_Reward/termination_penalty': [], 'Episode_Reward/feet_slide': [], 'Episode_Reward/joint_deviation_hip': [], 'Episode_Reward/joint_deviation_arms': [], 'Episode_Reward/joint_deviation_torso': [], 'Metrics/base_velocity/error_vel_xy': [], 'Metrics/base_velocity/error_vel_yaw': [], 'Episode_Termination/time_out': [], 'Episode_Termination/base_contact': []}
    
    demo_h1 = H1FlatDemo()
    obs, _ = demo_h1.env.reset()
    while simulation_app.is_running():
        demo_h1.update_selected_object()        
        with torch.inference_mode():
            action = demo_h1.policy(obs)            
            obs, reward, done, info = demo_h1.env.step(action)
            if done[0] != 0: # reset
                print("termination log:", info.get('log', {}))
                if 'log' in info:
                    for k, v in info['log'].items():
                        reward_split.setdefault(k, []).append(v)
                        
#                for k in info['log'].keys():
#                    reward_split[k].append(info['log'][k])
                print(reward_split)
                rewards.append(reward_list)
                reward_list = []
                print(f"Episode {len(rewards)} over!")
                if len(rewards) == 1:
                    break
            else:
                reward_list.append(reward)

    results = {'rewards': rewards, 'individual_rewards': reward_split}
    with open("scripts/UMBC/logs/metrics/expert_stairway.pkl", "wb") as f:  # "wb" = write binary
        pickle.dump(results, f)
    print('===============')
    print('Expert - Stairway Terrain')
    print('===============')
    print('Length of episode: ', [len(reward) for reward in rewards])
    print('Average length of episode: ', sum([len(reward) for reward in rewards])/len(rewards))
    print('Number of episode: ', len(rewards))
    print('Per episode reward: ', [sum(reward) for reward in rewards])
    print('Average reward: ', sum([sum(reward) for reward in rewards])/len(rewards))
    print('Total reward: ', sum([sum(reward) for reward in rewards]), '\n\n')

if __name__ == "__main__":
    main()
    simulation_app.close()
