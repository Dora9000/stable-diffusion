import os

SCALE = 7.5
H = 512
W = 512
C = 4  # latent channels
F = 8  # downsampling factor
DDIM_STEPS = 50
DDIM_ETA = 0.0  # ddim eta (eta=0.0 corresponds to deterministic sampling
SKIP_SAVE = False
OUTDIR = "outputs/txt2img-samples"

RABBITMQ_LOGIN = os.getenv("RABBITMQ_LOGIN", "admin")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "87.197.111.68")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", 41077)
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "password")
