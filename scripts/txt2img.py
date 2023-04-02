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
from ldm.models.lgp_train import LGPDataset
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from scripts.img2img import load_img

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


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


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    # safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    # x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    # assert x_checked_image.shape[0] == len(has_nsfw_concept)
    # for i in range(len(has_nsfw_concept)):
    #     if has_nsfw_concept[i]:
    #         x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_image, False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=randrange(1000),
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # loaded input image of size (375, 393) from inputs/water.jpg
    # torch.Size([1, 4, 48, 40])

    # loaded input image of size (901, 442) from inputs/test.png
    # torch.Size([1, 4, 48, 112])

    # model_path = "models/lgp/model.pt"
    model_path = "/workspace/stable-diffusion/models/lgp/my_model/model3" + '.pt'

    guiding_model = latent_guidance_predictor(output_dim=4, input_dim=9320, num_encodings=9).to(device)  # 7080
    # guiding_model.init_weights()

    checkpoint = torch.load(model_path, map_location=device)
    guiding_model.load_state_dict(checkpoint['model_state_dict'])
    guiding_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    guiding_model.eval()

    # if opt.plms, else: sampler = DDIMSampler(model)
    sampler = PLMSSampler(model, guiding_model=guiding_model)


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


    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    from torch.utils.data import Dataset, DataLoader



    dataset = LGPDataset(dataset_dir="/workspace/datasets/workspace/dataset/imagenet_images/", edge_maps_dir="/workspace/datasets/workspace/dataset/cleared/")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    ii = 0

    for image_path, edge_map_path, prompt in iter(dataloader):

        image_path, *_ = image_path
        edge_map_path, *_ = edge_map_path
        prompt, *_ = prompt

        ii += 1

        init_image = load_img(image_path).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=opt.n_samples)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent spac

        edge_image = load_img(edge_map_path).to(device)
        edge_image = repeat(edge_image, '1 ... -> b ...', b=opt.n_samples)
        edge_latent = model.get_first_stage_encoding(model.encode_first_stage(edge_image))  # move to latent spac

        data = [batch_size * [prompt]]

        precision_scope = autocast if opt.precision=="autocast" else nullcontext

        # DEBUG
        if ii % 10 == 0:

            print(f'============== log {ii} ================= ')
            sampler.guiding_model.is_log = True



        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    # all_samples = list()
                    for _ in range(opt.n_iter):
                        for prompts in data:
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                             conditioning=c,
                                                             batch_size=opt.n_samples,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=opt.scale,
                                                             unconditional_conditioning=uc,
                                                             eta=opt.ddim_eta,
                                                             x_T=start_code,
                                                             sketch_img=edge_latent,
                                                             orig=init_latent)
                            xs = sampler.guiding_model.loss
                            print(sum(xs) / len(xs))
                            sampler.guiding_model.loss = []
                #         x_samples_ddim = model.decode_first_stage(samples_ddim)
                #         x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                #         x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                #
                #         x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                #
                #         x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                #
                #         if not opt.skip_save:
                #             for x_sample in x_checked_image_torch:
                #                 x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                #                 img = Image.fromarray(x_sample.astype(np.uint8))
                #                 img = put_watermark(img, wm_encoder)
                #                 img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                #                 base_count += 1
                #
                #         if not opt.skip_grid:
                #             all_samples.append(x_checked_image_torch)
                #
                # if not opt.skip_grid:
                #     # additionally, save as grid
                #     grid = torch.stack(all_samples, 0)
                #     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                #     grid = make_grid(grid, nrow=n_rows)
                #
                #     # to image
                #     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                #     img = Image.fromarray(grid.astype(np.uint8))
                #     img = put_watermark(img, wm_encoder)
                #     img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                #     grid_count += 1
                #
                # toc = time.time()

        if ii % 4 == 0:
            print(f'============== step {ii} ================= ')

            print('saving to:', model_path)
            torch.save({
                'model_state_dict': sampler.guiding_model.state_dict(),
                'optimizer_state_dict': sampler.guiding_model.optimizer.state_dict(),
                # 'loss': loss,
            }, model_path)




        # DEBUG
        if sampler.guiding_model.log_img is not None:
            sampler.guiding_model.is_log = False
            precision_scope = autocast if opt.precision == "autocast" else nullcontext

            with torch.no_grad():
                with precision_scope("cuda"):

                    out = model.decode_first_stage(sampler.guiding_model.log_img)
                    sampler.guiding_model.log_img = None
                    out = torch.clamp((out + 1.0) / 2.0, min=0.0, max=1.0)
                    out = out.cpu().permute(0, 2, 3, 1).numpy()

                    out, _ = check_safety(out)

                    out = torch.from_numpy(out).permute(0, 3, 1, 2)

                    if not opt.skip_save:
                        for x_sample in out:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img = put_watermark(img, wm_encoder)
                            print(image_path)
                            img.save(os.path.join(sample_path, f"{base_count:05}-{image_path.split('/')[-1].split('.')[0]}.png"))
                            base_count += 1


    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
