import imageio
import argparse
import os
import torch
from model import VAE
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


to_tensor = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Resize((244,244)),
                ])

to_pil = transforms.ToPILImage()

def image_to_latents(model, image_path):

    image = Image.open(image_path).convert('RGB')
    image = to_tensor(image)
    image = image.unsqueeze(0)

    latents, _, _ = model.encoder(image)

    return latents

def latents_to_image(model, latents):

    image = model.decoder(latents)
    image = to_pil(image[0])

    return image
    
def latent_image_path(model, image_list, num_steps=100):
    
    for index in tqdm(range(len(image_list) - 1)):
        A = image_to_latents(model, image_list[index])
        B = image_to_latents(model, image_list[index+1])

        step = (B - A) / num_steps

        latents = A
        images = []
        for _ in range(num_steps):

            latents += step
            output = model.decoder(latents)

            image = to_pil(output.squeeze())
            images.append(image)

    imageio.mimsave('movie.gif', images)

def latent_random_path(model, num_samples, num_steps=100):
    
    A = torch.randn(1,256,1,1)
    for _ in tqdm(range(num_samples)):
        B = torch.randn(1,256,1,1)

        step = (B - A) / num_steps

        latents = A
        images = []
        for _ in range(num_steps):

            latents += step
            output = model.decoder(latents)
            image = to_pil(output.squeeze())
            images.append(image)

        A = B

    imageio.mimsave('movie.gif', images)




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
    model = VAE(channel_in=3, latent_channels=256).to(device)
    model.device = device
    weights = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(weights)

    # image_list = [
    #     'images/im1.jpg',
    #     'images/im2.jpg',
    #     'images/im3.jpg',
    #     'images/im4.jpg'
    # ]
    # latent_image_path(model, image_list)

    latent_random_path(model, num_samples=10)

    # latents = image_to_latents(model, 'images/im1.jpg')
    # image = latents_to_image(model, latents)
    # image.show()

