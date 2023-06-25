import imageio
import argparse
import os
import torch
from model import VAE
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


to_tensor = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((244,244)),
                ])

to_pil = transforms.ToPILImage()

def image_to_latents(model, image_path):

    image = Image.open(image_path).convert('RGB')
    image = to_tensor(image)
    image = image.unsqueeze(0)

    latents, _, _ = model.encoder(image.to(model.device))

    return latents

def latents_to_image(model, latents):

    image = model.decoder(latents.to(model.device))
    image = to_pil(image[0])

    return image
    
def latent_image_path(model, image_list, num_steps=100):

    # add the first image to the end of the image list so that the gif will endlessly loop
    image_list.append(image_list[0])
    images = []
    for index in tqdm(range(len(image_list) - 1)):
        A = image_to_latents(model, image_list[index])
        B = image_to_latents(model, image_list[index+1])

        step = (B - A) / num_steps

        latents = A
        for _ in range(num_steps):

            latents += step
            output = model.decoder(latents.to(model.device))

            image = to_pil(output.squeeze())
            images.append(image)

    imageio.mimsave('fungai.gif', images)

if __name__ == "__main__":

    # initialise argument parser
    parser = argparse.ArgumentParser(description='This file is for generating ML datasets')

    # set arguments
    parser.add_argument('-w','--weights_path', type=str, required=True,
                        help='Weights of the model to load and analyse')

    # extract arguments
    args = parser.parse_args()

    # define the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # define the model
    model = VAE(channel_in=3, latent_channels=512, ch=80).to(device)
    model.device = device
    weights = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(weights)

    image_list = [f'images/{img}' for img in os.listdir('images')]

    latent_image_path(model, image_list, num_steps=70)
