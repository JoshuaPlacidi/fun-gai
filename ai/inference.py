def encode(model_weights_path, image_path):
    # TODO takes a image_path and returns the laten representation

    return

def decode(model_weights_path, latents):
    # TODO takes a latent representation and returns an image

    return

def generate_and_store_latents(model_weights_path, image_paths: list[str], save_path: str):
    # TODO takes a list of image paths, passes them through the encoder, compresses the latents to 2D and stores at save_path