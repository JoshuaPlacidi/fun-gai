import imageio
import argparse
import os
import torch
from model import VAE
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import random
import cv2
import numpy as np


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
    
def latent_image_path(model, image_list, num_steps=100, num_waits=20):
    model.eval()
    # add the first image to the end of the image list so that the gif will endlessly loop
    image_list.append(image_list[0])

    videodims = (256,256)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    video = cv2.VideoWriter("fungai.mp4",fourcc, 60,videodims)

    for index in tqdm(range(len(image_list) - 1)):
        A = image_to_latents(model, image_list[index])
        B = image_to_latents(model, image_list[index+1])

        A_image = to_pil(model.decoder(A.to(model.device)).squeeze())
        
        for _ in range(num_waits):
            video.write(cv2.cvtColor(np.array(A_image), cv2.COLOR_RGB2BGR))

        step = (B - A) / num_steps
        latents = A

        for s in range(num_steps-1):

            latents += step
            output = model.decoder(latents.to(model.device))

            image = to_pil(output.squeeze())

            video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    video.release()

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
    random.shuffle(image_list)

    latent_image_path(model, image_list, num_steps=50)
