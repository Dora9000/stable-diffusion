import os
import asyncio
import json
from random import randrange
import torch
from omegaconf import OmegaConf
from imwatermark import WatermarkEncoder
from pytorch_lightning import seed_everything
from ldm.models.diffusion.ddpm import get_features
from ldm.models.latent_guidance_predictor import latent_guidance_predictor
from ldm.util import instantiate_from_config

from rabbitmq.consumer import GenerationConsumer
from rabbitmq.settings import RabbitmqData
from rabbitmq.storage import GenerationStorage
import settings


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    # if "global_step" in pl_sd:
    #     print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # if len(m) > 0 and verbose:
    #     print("missing keys:")
    #     print(m)
    # if len(u) > 0 and verbose:
    #     print("unexpected keys:")
    #     print(u)

    model.cuda()
    model.eval()
    return model


def initialize():
    seed_everything(randrange(1000))

    config = "configs/stable-diffusion/v1-inference.yaml"
    ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"

    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    LGP_path = "/workspace/stable-diffusion/models/lgp/my_model/model2.pt"

    guiding_model = latent_guidance_predictor(output_dim=4, input_dim=9320, num_encodings=9).to(device)
    checkpoint = torch.load(LGP_path, map_location=device)
    guiding_model.load_state_dict(checkpoint['model_state_dict'])
    guiding_model.eval()

    model.model.diffusion_model.input_blocks[2].register_forward_hook(get_features('input_blocks_2'))
    model.model.diffusion_model.input_blocks[4].register_forward_hook(get_features('input_blocks_4'))
    model.model.diffusion_model.input_blocks[8].register_forward_hook(get_features('input_blocks_8'))

    model.model.diffusion_model.middle_block[0].register_forward_hook(get_features('middle_block_0'))
    model.model.diffusion_model.middle_block[1].register_forward_hook(get_features('middle_block_1'))
    model.model.diffusion_model.middle_block[2].register_forward_hook(get_features('middle_block_2'))

    model.model.diffusion_model.output_blocks[2].register_forward_hook(get_features('output_blocks_2'))
    model.model.diffusion_model.output_blocks[4].register_forward_hook(get_features('output_blocks_4'))
    model.model.diffusion_model.output_blocks[8].register_forward_hook(get_features('output_blocks_8'))

    os.makedirs(settings.OUTDIR, exist_ok=True)

    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
    
    print("initialized")
    return model, guiding_model
    

async def on_message(service: GenerationConsumer, message, model, guiding_model) -> None:
    body = json.loads(message.body)
    message_id = body.pop("message_id")
    print(f"Received message with message_id={message_id}")
    await service.react_message(message=body, model=model, guiding_model=guiding_model)
    print(f"Message with message_id={message_id} has been processed")
    await message.ack()
    print(f"Message with message_id={message_id} was confirmed")


async def main() -> None:
    model, guiding_model = initialize()

    connection = await GenerationStorage().connection()

    service = GenerationConsumer()

    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(RabbitmqData.generation.default_queue)

        async with queue.iterator(no_ack=False) as queue_iter:
            async for message in queue_iter:
                await on_message(service=service, message=message, model=model, guiding_model=guiding_model)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())


