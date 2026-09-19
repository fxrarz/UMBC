# perturbation
#python3 scripts/UMBC/3.test/unified_stairway_perturbation.py --push_force 50 --push_interval 10 --push_duration 10
# no perturbation
#python3 scripts/UMBC/3.test/unified_stairway_perturbation.py --push_force 0 --push_interval 10 --push_duration 10


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

# --- perturbation / fall-count args ---
parser.add_argument("--push_force", type=float, default=200.0,
                     help="Magnitude of external push force (N) applied to the robot base.")
parser.add_argument("--push_interval", type=int, default=100,
                     help="Number of simulation steps between pushes.")
parser.add_argument("--push_duration", type=int, default=5,
                     help="Number of simulation steps the push force is held for before clearing.")
parser.add_argument("--num_episodes", type=int, default=20,
                     help="Number of episodes to run before reporting fall statistics.")
parser.add_argument("--push_body_name", type=str, default=None,
                     help="Regex matching the body to push (e.g. 'torso_link' or 'pelvis'). "
                          "If omitted, pushes body index 0 (the root body).")
parser.add_argument("--fall_key", type=str, default="Episode_Termination/base_contact",
                     help="Key in info['log'] used to identify a fall. Check the printed "
                          "termination log on your first run to confirm this key exists "
                          "for the stairway task.")

args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import torch.nn as nn
import math
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
import json

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

        sub_terrains = env_cfg.scene.terrain.terrain_generator.sub_terrains
        for terrain_key in ["pyramid_stairs", "pyramid_stairs_inv"]:
            if terrain_key in sub_terrains:
                sub_terrains[terrain_key].border_width = DESIRED_SUB_TERRAIN_BORDER_WIDTH

        # ---------------------------------------------------------
        # LOCAL OVERRIDE: Randomize Spawn Locations on Reset
        # ---------------------------------------------------------
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base"):
            env_cfg.events.reset_base.params["pose_range"] = {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14)
            }

        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using {device} device")
        self.policy = LocomotionPolicy().to(device)
        self.policy.load_state_dict(torch.load('scripts/UMBC/logs/models/umbc.pth', weights_only=False))
        self.policy.eval()
        self.create_camera()
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)

        # --- perturbation setup ---
        self.robot = self.env.unwrapped.scene["robot"]
        if args_cli.push_body_name is not None:
            body_ids, body_names = self.robot.find_bodies(args_cli.push_body_name)
            if len(body_ids) == 0:
                raise ValueError(
                    f"No bodies matched '{args_cli.push_body_name}'. "
                    f"Available bodies: {self.robot.body_names}"
                )
            self.push_body_ids = body_ids
            print(f"Pushing bodies: {body_names}")
        else:
            self.push_body_ids = [0]
            print(f"Pushing default root body: {self.robot.body_names[0]} "
                  f"(pass --push_body_name to target a different link)")

    def apply_push(self, force_vec):
        num_bodies = len(self.push_body_ids)
        forces = torch.zeros(self.env.unwrapped.num_envs, num_bodies, 3, device=self.device)
        torques = torch.zeros_like(forces)
        forces[:, :, :] = force_vec
        self.robot.set_external_force_and_torque(forces, torques, body_ids=self.push_body_ids)

    def clear_push(self):
        num_bodies = len(self.push_body_ids)
        forces = torch.zeros(self.env.unwrapped.num_envs, num_bodies, 3, device=self.device)
        torques = torch.zeros_like(forces)
        self.robot.set_external_force_and_torque(forces, torques, body_ids=self.push_body_ids)

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
    reward_split = {}  # built dynamically from info['log'] on first episode end

    demo_h1 = H1FlatDemo()
    obs, _ = demo_h1.env.reset()
    print("root height right after reset:", demo_h1.env.unwrapped.scene["robot"].data.root_pos_w[:, 2].item())

    number_envs = 1

    with open("scripts/UMBC/logs/models/terrain_mapping.json") as f:
        terrain_to_index = json.load(f)
    terrain_id = terrain_to_index["stairway"]

    terrain = torch.full((number_envs,), terrain_id, device='cuda:0', dtype=torch.long)

    # --- perturbation / fall tracking state ---
    step_count = 0
    push_active = False
    push_end_step = -1
    fall_count = 0
    fall_flags = []

    while simulation_app.is_running():
        demo_h1.update_selected_object()

        if not push_active and step_count > 0 and step_count % args_cli.push_interval == 0:
            angle = torch.rand(1).item() * 2 * math.pi
            force_vec = torch.tensor(
                [args_cli.push_force * math.cos(angle),
                 args_cli.push_force * math.sin(angle),
                 0.0],
                device=demo_h1.device,
            )
            demo_h1.apply_push(force_vec)
            push_active = True
            push_end_step = step_count + args_cli.push_duration
            print(f"[step {step_count}] Applying push: {args_cli.push_force} N at angle {math.degrees(angle):.1f} deg")

        if push_active and step_count >= push_end_step:
            demo_h1.clear_push()
            push_active = False

        with torch.inference_mode():
            height = obs[:, 69:]
            obs_proprio = obs[:, :69]
            action = demo_h1.policy(obs_proprio, height, terrain)
            obs, reward, done, info = demo_h1.env.step(action)

            if done[0] != 0:  # reset
                print("root height right after reset:", demo_h1.env.unwrapped.scene["robot"].data.root_pos_w[:, 2].item())
                print("termination log:", info['log'])
                for k, v in info['log'].items():
                    reward_split.setdefault(k, []).append(v)

                fell = float(info['log'].get(args_cli.fall_key, 0.0)) > 0
                fall_flags.append(fell)
                if fell:
                    fall_count += 1

                rewards.append(reward_list)
                print(f"Episode {len(rewards)} over! Fell: {fell}")

                if len(rewards) >= args_cli.num_episodes:
                    break
                reward_list = []

                demo_h1.clear_push()
                push_active = False
            else:
                reward_list.append(reward[0].item())

        step_count += 1

    results = {
        'rewards': rewards,
        'individual_rewards': reward_split,
        'fall_count': fall_count,
        'fall_flags': fall_flags,
        'push_force': args_cli.push_force,
        'push_interval': args_cli.push_interval,
        'push_duration': args_cli.push_duration,
    }
    os.makedirs("scripts/UMBC/logs/metrics", exist_ok=True)
 
    if args_cli.push_force == 0:
        with open("scripts/UMBC/logs/metrics/test_umbc_stairway_noperturbation.pkl", "wb") as f:
            pickle.dump(results, f)
    else:
        with open("scripts/UMBC/logs/metrics/test_umbc_stairway_perturbation.pkl", "wb") as f:
            pickle.dump(results, f)


    print('===============')
    print('UMBC Stairway Terrain under perturbation')
    print('===============')
    print('Push force (N): ', args_cli.push_force)
    print('Push interval (steps): ', args_cli.push_interval)
    print('Push duration (steps): ', args_cli.push_duration)
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
