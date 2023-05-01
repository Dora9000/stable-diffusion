import os
import cv2
import torch
import numpy as np
from PIL import Image
from imwatermark import WatermarkEncoder
from einops import rearrange
from torch import autocast
from einops import repeat

from ldm.models.diffusion.plms import PLMSSampler
from rabbitmq.producer import StatusProducer
from scripts.img2img import load_img
import settings


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


class GenerationConsumer:

    @classmethod
    async def react_message(cls, model, guiding_model, message: dict) -> None:
        file_name = message["file_name"]
        file_name = f"inputs/{file_name}"
        file_id = message["file_id"]
        prompt = message["prompt"]
        prompt = [prompt]

        init_k = float(message["init_k"])
        grad_k = float(message["grad_k"])

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        assert os.path.isfile(file_name)
        init_image = load_img(file_name).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=1)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent spac

        sampler = PLMSSampler(model, guiding_model=guiding_model, sketch_target=init_latent)

        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():

                    uc = None
                    if settings.SCALE != 1.0:
                        uc = model.get_learned_conditioning([""])

                    c = model.get_learned_conditioning(prompt)
                    shape = [settings.C, settings.H // settings.F, settings.W // settings.F]
                    samples_ddim, _ = await sampler.sample(
                        S=settings.DDIM_STEPS,
                        conditioning=c,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=settings.SCALE,
                        unconditional_conditioning=uc,
                        eta=settings.DDIM_ETA,
                        x_T=None,
                        sketch_img=init_latent,
                        init_k=init_k,
                        grad_k=grad_k,
                        queue_data={
                            'reply_chat_id': message["reply_chat_id"],
                            'reply_message_id': message["reply_message_id"],
                        }
                    )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                    if not settings.SKIP_SAVE:
                        for x_sample in x_checked_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img = put_watermark(img, wm_encoder)
                            img.save(os.path.join(os.path.join(settings.OUTDIR, "samples"), f"{file_id}-generated.png"))

        await StatusProducer().send(
            data={
                "percent": 100,
                "file_name": f"{file_id}-generated.png",
                'reply_chat_id': message["reply_chat_id"],
                'reply_message_id': message["reply_message_id"],
            }
        )
        print('done!')

