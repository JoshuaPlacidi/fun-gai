import torch
from torchvision import transforms
import cv2
import numpy as np
from model import VAE
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path: str):
    """
    Loads an image from a path and returns it as a tensor
    """
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((64, 64)), # Resize images to the appropriate size
        transforms.ToTensor(), # Convert to tensor
    ])
    img = transform(img).unsqueeze(0)
    return img

def encode(image_path: list[str], model_weights_path: str = None):
    """
    Takes a image_path and returns the latent representation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(channel_in=3, latent_channels=64)
    if model_weights_path is not None:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    # When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector 
    # and log_var will be None
    model.eval()

    # Load image and convert to tensor
    img = load_image(image_path).to(device)

    with torch.no_grad():
        latents, _, _ = model.encoder(img)
    
    return latents

def decode(latents, model_weights_path: str = None):
    """
    Takes a latent representation and returns an image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(channel_in=3, latent_channels=64)
    if model_weights_path is not None:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    # model = model.to(device)
    model.eval()

    latents = latents.to(device)
    print(latents.shape)
    # Cast latents to double
    latents = latents.float()
    print(latents.dtype)

    with torch.no_grad():
        img_recon = model.decoder(latents)

    # print(img_recon)

    # img_recon = img_recon.clamp(0, 1)

    return img_recon


def generate_and_store_latents(image_paths: list[str], save_path: str, model_weights_path: str = None):
    """
    Takes a list of image paths, passes them through the encoder, compresses the latents to 2D and stores at save_path
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(channel_in=3, latent_channels=64)
    if model_weights_path is not None:
        model.load_state_dict(torch.load(model_weights_path))
    model = model.to(device)
    model.eval()

    latents = []
    for image_path in image_paths:
        # Load image and convert to tensor
        img = load_image(image_path).to(device)
        encoding, _, _ = model.encoder(img)
        latents.append(encoding.detach().numpy())

    latents = np.concatenate(latents, axis=0)
    np.save(save_path, latents)

if __name__ == "__main__":
    # Encode
    latents = encode("shrooms_1.jpeg", model_weights_path="3_93_model.pth")
    print("Latents shape: ", latents.shape)

    # Decode
    img_recon = decode(latents, model_weights_path="3_93_model.pth")
    print("Reconstructed image shape: ", img_recon.shape)

    # Get rid of batch dimension
    img_recon = img_recon.squeeze()
    
    # Display reconstructed image
    # Convert to range [0, 255]
    T = transforms.Compose([
        transforms.ConvertImageDtype(torch.uint8),
        transforms.ToPILImage(),
    ])
    img = T(img_recon)
    img.show()
    
    # Save mutliple latents
    generate_and_store_latents(["shrooms_1.jpeg", "shrooms_2.jpeg"], "encodings/latents.npy")

