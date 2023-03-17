from typing import List
import math
import torch
from torch import nn


class latent_guidance_predictor(nn.Module):
    def __init__(self, output_dim, input_dim, num_encodings):
        super(latent_guidance_predictor, self).__init__()
        self.num_encodings = num_encodings

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, output_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.loss = []
        self.log_img = None
        self.is_log = False
        self.ts = 0

    def forward(self, x, t):
        # Concatenate input pixels with noise level t and positional encodings
        _t = t.transpose(1,3)[:1]
        pos_encoding = [torch.sin(2 * math.pi * _t * (2 **-l)) for l in range(self.num_encodings)]
        pos_encoding = torch.cat(pos_encoding, dim=-1)
        #torch.Size([1, 64, 64, 9280]) torch.Size([1, 64, 64, 4]) torch.Size([1, 64, 64, 36])
        x = torch.cat((x, _t, pos_encoding), dim=-1)
        x = x.flatten(start_dim=0, end_dim=2)
        #torch.Size([4096, 9320])
        return self.layers(x)


    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


def resize_and_concatenate(activations: List[torch.Tensor], reference):
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = reference.shape[2:]
    resized_activations = []

    # activation - input_blocks_2 -  (2, 320, 64, 64)
    # activation - input_blocks_4 -  (2, 640, 32, 32)
    # activation - input_blocks_8 -  (2, 1280, 16, 16)
    # activation - middle_block_0 -  (2, 1280, 8, 8)
    # activation - middle_block_1 -  (2, 1280, 8, 8)
    # activation - middle_block_2 -  (2, 1280, 8, 8)
    # activation - output_blocks_2 -  (2, 1280, 16, 16)
    # activation - output_blocks_4 -  (2, 1280, 16, 16)
    # activation - output_blocks_8 -  (2, 640, 64, 64)

    for acts in activations:
        # torch.Size([2, 1280, 8, 8])
        acts = nn.functional.interpolate(acts, size=size, mode="bilinear")
        # torch.Size([2, 1280, 64, 64])
        acts = acts[:1]
        # torch.Size([1, 1280, 64, 64])
        acts = acts.transpose(1, 3)
        # torch.Size([1, 64, 64, 1280])
        resized_activations.append(acts)

    a = torch.cat(resized_activations, dim=3)

    # torch.Size([1, 64, 64, 9280])
    return a
