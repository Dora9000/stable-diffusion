import argparse, os, sys, glob
from random import randrange

import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from einops import repeat
from ldm.models.diffusion.ddpm import get_features
from ldm.models.latent_guidance_predictor import latent_guidance_predictor
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from scripts.img2img import load_img

from fastapi import FastAPI


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model



def initialize():
    seed_everything(123)
    
    config = "configs/stable-diffusion/v1-inference.yaml"
    ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"

    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # loaded input image of size (375, 393) from inputs/water.jpg
    # torch.Size([1, 4, 48, 40])

    # loaded input image of size (901, 442) from inputs/test.png
    # torch.Size([1, 4, 48, 112])

    LGP_path = "/workspace/stable-diffusion/models/lgp/my_model/model2.pt"

    guiding_model = latent_guidance_predictor(output_dim=4, input_dim=9320, num_encodings=9).to(device)
    checkpoint = torch.load(LGP_path, map_location=device)
    guiding_model.load_state_dict(checkpoint['model_state_dict'])
    guiding_model.eval()


    # The denoising model’s features are taken from 9 different layers across the network:
    # input block - layers 2, 4, 8,
    # middle block - layers 0, 1, 2,
    # output block - layers 2, 4, 8.

    model.model.diffusion_model.input_blocks[2].register_forward_hook(get_features('input_blocks_2'))
    model.model.diffusion_model.input_blocks[4].register_forward_hook(get_features('input_blocks_4'))
    model.model.diffusion_model.input_blocks[8].register_forward_hook(get_features('input_blocks_8'))

    model.model.diffusion_model.middle_block[0].register_forward_hook(get_features('middle_block_0'))
    model.model.diffusion_model.middle_block[1].register_forward_hook(get_features('middle_block_1'))
    model.model.diffusion_model.middle_block[2].register_forward_hook(get_features('middle_block_2'))

    model.model.diffusion_model.output_blocks[2].register_forward_hook(get_features('output_blocks_2'))
    model.model.diffusion_model.output_blocks[4].register_forward_hook(get_features('output_blocks_4'))
    model.model.diffusion_model.output_blocks[8].register_forward_hook(get_features('output_blocks_8'))
    
    
    outpath = "outputs/txt2img-samples"
    os.makedirs(outpath, exist_ok=True)

    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
    
    print("initialized")
    
    

if __name__ == "__main__":
    initialize()

    app = FastAPI()
#     api = create_api(app)

#     print(f"Startup")
#     api.launch(server_name="127.0.0.1", port=7861)
    


