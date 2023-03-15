import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class LGPDataset(Dataset):
    def __init__(self, dataset_dir, edge_maps_dir):
        self.dataset_dir = dataset_dir
        self.edge_maps_dir = edge_maps_dir
        self.image_paths = []

        # Iterate through all subfolders in dataset_dir
        for subfolder in os.listdir(self.dataset_dir):
            subfolder_path = os.path.join(self.dataset_dir, subfolder)
            caption = subfolder

            # Iterate through all images in the subfolder
            for image_file in os.listdir(subfolder_path):
                if image_file.endswith('.jpg'):
                    image_path = os.path.join(subfolder_path, image_file)
                    image_unsure = Image.open(image_path)
                    if image_unsure.mode != "RGB":
                        continue

                    edge_map_file = image_file.replace('.jpg', '.png')
                    edge_map_path_unsure = os.path.join(self.edge_maps_dir, edge_map_file)
                    if os.path.exists(edge_map_path_unsure):
                        edge_map_path = edge_map_path_unsure
                        self.image_paths.append((image_path, edge_map_path, caption))
                    else:
                        continue

    def __getitem__(self, index):
        image_path, edge_map_path, caption = self.image_paths[index]
        return image_path, edge_map_path, caption

        # image = Image.open(image_path)
        # image = image.resize((512, 512))
        #
        # edge_map = Image.open(edge_map_path)
        # rgb_edge_map = Image.merge('RGB', (edge_map, edge_map, edge_map))
        # rgb_edge_map = rgb_edge_map.resize((512, 512))
        #
        # encoded_edge_map = img_to_latents(rgb_edge_map)
        # encoded_edge_map = encoded_edge_map.transpose(1, 3)
        #
        # encoded_image = img_to_latents(image).to(device)
        # noisy_image, noise_level, timesteps = noisy_latent(encoded_image, noise_scheduler)
        # text_input = tokenizer([caption], padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
        #                        return_tensors="pt")
        # with torch.no_grad():
        #     text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        # max_length = text_input.input_ids.shape[-1]
        # uncond_input = tokenizer(
        #     [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        # with torch.no_grad():
        #     uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        # text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        # features = extract_features(noisy_image, blocks, text_embeddings, timesteps)
        # features = resize_and_concatenate(features, encoded_image)
        # noise_level = noise_level.transpose(1, 3)

        # return features, encoded_edge_map, noise_level

    def __len__(self):
        return len(self.image_paths)


# def img_to_latents(self, img: Image):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     init_image = load_img(opt.init_img).to(device)
#     init_image = repeat(init_image, '1 ... -> b ...', b=opt.n_samples)
#     init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent spac
#
#     np_img = (np.array(img).astype(np.float32) / 255.0) * 2.0 - 1.0
#     np_img = np_img[None].transpose(0, 3, 1, 2)
#     torch_img = torch.from_numpy(np_img)
#     generator = torch.Generator(device).manual_seed(0)
#     latents = self.vae.encode(torch_img.to(self.vae.dtype).to(device)).latent_dist.sample(generator=generator)
#     latents = latents * 0.18215
#     return latents
#
#
# a = LGPDataset(
#     dataset_dir="/workspace/dataset/imagenet_images/",
#     edge_maps_dir="/workspace/dataset/cleared/",
# )
# print(len(a))
