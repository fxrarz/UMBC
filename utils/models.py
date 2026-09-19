import torch
import torch.nn as nn

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
