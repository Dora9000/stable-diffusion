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



# def save_tensors(module: nn.Module, features, name: str):
#     """ Process and save activations in the module. """
#     if type(features) in [list, tuple]:
#         features = [f.detach().float() for f in features if f is not None and isinstance(f, torch.Tensor)]
#         setattr(module, name, features)
#     elif isinstance(features, dict):
#         features = {k: f.detach().float() for k, f in features.items()}
#         setattr(module, name, features)
#     else:
#         setattr(module, name, features.detach().float())
#
# def save_out_hook(self, inp, out):
#     save_tensors(self, out, 'activations')
#     return out

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

# def grad(pred_map, target):
#     with torch.enable_grad():
#         diff = pred_map - target
#         d = diff.detach().requires_grad_()
#         ((torch.linalg.vector_norm(d))**2).backward()
#
#     return d.grad


# def mlp_train(model, X, y):
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     loss_fn = nn.CrossEntropyLoss()
#
#     for epoch in range(epochs := 300):
#         model.train()
#
#         optimizer.zero_grad()
#
#         outputs = model(X)
#         loss = loss_fn(outputs, y)
#         loss.backward()
#         optimizer.step()
#
#         print('train_loss', loss.item())
#
#         model.eval()
#
#     return model
