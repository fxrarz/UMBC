# python3 UMBC/test/flat_umbc.py

import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
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
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg_PLAY
import pickle

#TASK = "Isaac-Velocity-Flat-H1-v0"
TASK = "Isaac-Velocity-Flat-H1-v0"
RL_LIBRARY = "rsl_rl"

class H1FlatDemo:
    def __init__(self):
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
        env_cfg = H1FlatEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1
        env_cfg.curriculum = None
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using {device} device")
        self.policy = LocomotionPolicy().to(device)
        self.policy.load_state_dict(torch.load('UMBC/logs/policy/UMBC_policy.pth', weights_only=False))
        self.policy.eval()
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

def main():
    rewards = []
    reward_list = []
    reward_split = {'Episode_Reward/track_lin_vel_xy_exp':[], 'Episode_Reward/track_ang_vel_z_exp': [], 'Episode_Reward/ang_vel_xy_l2': [], 'Episode_Reward/dof_torques_l2': [], 'Episode_Reward/dof_acc_l2': [], 'Episode_Reward/action_rate_l2': [], 'Episode_Reward/feet_air_time': [], 'Episode_Reward/flat_orientation_l2': [], 'Episode_Reward/dof_pos_limits': [], 'Episode_Reward/termination_penalty': [], 'Episode_Reward/feet_slide': [], 'Episode_Reward/joint_deviation_hip': [], 'Episode_Reward/joint_deviation_arms': [], 'Episode_Reward/joint_deviation_torso': [], 'Metrics/base_velocity/error_vel_xy': [], 'Metrics/base_velocity/error_vel_yaw': [], 'Episode_Termination/time_out': [], 'Episode_Termination/base_contact': []}
    
    demo_h1 = H1FlatDemo()
    obs, _ = demo_h1.env.reset()
    
    number_envs = 1
    terrain_id = 0
    terrain = torch.full((number_envs,), terrain_id, device='cuda:0', dtype=torch.long)
    height = torch.full((number_envs, 187), 0, device='cuda:0', dtype=torch.float32)

    while simulation_app.is_running():
        demo_h1.update_selected_object()        
        with torch.inference_mode():
            action = demo_h1.policy(obs, height, terrain)            
            obs, reward, done, info = demo_h1.env.step(action)
            if done[0] != 0: # reset
                for k in info['log'].keys():
                    reward_split[k].append(info['log'][k])
                print(reward_split)
                if len(rewards) == 1:
                    break
                rewards.append(reward_list)
                reward_list = []
                print(f"Episode {len(rewards)} over!")
            else:
                reward_list.append(reward)
            #obs[:, 9:13] = demo_h1.commands

    results = {'rewards': rewards, 'individual_rewards': reward_split}
    with open("UMBC/logs/results/flat_policy.pkl", "wb") as f:
        pickle.dump(results, f)
    print('===============')
    print('Flat Terrain')
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
