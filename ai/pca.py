from inference import encode
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import pickle as pk

def pca_encode_images(image_folder: str, model_weights_path: str = None, n_components: int = 2):
    """
    Takes a directory of images, encodes them using a VAE, then applies PCA to the encodings.
    """
    # Make sure we're working with an absolute path
    image_folder = os.path.abspath(image_folder)

    # Create an empty list to hold the final data and labels
    all_latents = []
    labels = []

    # Traverse through the subfolders
    for root, _, files in os.walk(image_folder):
        # The subfolder name will serve as the class label
        class_label = os.path.basename(root)

        if class_label == 'Mushrooms':
            continue

        print("Encoding Species ", class_label, '...')

        # Process all the files in the subfolder
        for filename in tqdm(files):

            # Create the full path to the file
            image_path = os.path.join(root, filename)

            # Skip the file if it's not an image
            if not (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')):
                continue
            
            # Get the latent representation of the image
            latent = encode(image_path, model_weights_path)

            # Convert the latent representation to a numpy array and reshape it to 2D
            latent = np.array(latent)
            latent = latent.reshape(latent.shape[0], -1)

            # Add the latent vector and its label to the lists
            all_latents.append(latent)
            labels.append(class_label)

    # Convert the latents list to a numpy array
    all_latents = np.concatenate(all_latents, axis=0)

    # Create a PCA object
    pca = PCA(n_components=n_components)

    # Fit the PCA model to the latents and apply the dimensionality reduction
    pca_latents = pca.fit_transform(all_latents)

    # Save to be used later to invert the pca
    pk.dump(pca, open("pca.pkl","wb"))

    # Create an empty list to hold the final data
    data = []

    # Stores latents per species
    classes = np.unique(labels)
    latents_per_species = {c: [] for c in classes}

    # Combine the reduced latent vectors with their corresponding labels
    for i in tqdm(range(len(labels))):
        data.append([pca_latents[i, 0].item(), pca_latents[i, 1].item(), labels[i]])
        latents_per_species[labels[i]].append([pca_latents[i, 0].item(), pca_latents[i, 1].item()])
    
    # Comupte centroids per species and dump them in a file
    centroids_per_species = {}
    for c in classes:
        centroids_per_species[c] = np.mean(latents_per_species[c], axis=0).tolist()
    with open("centroids.json", "w") as f:
        json.dump(centroids_per_species, f)    

    return data

def save_json(data, filename):
    """
    Save a Python dict to a JSON file
    """
    with open(filename, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    # Perform PCA on the encoded images and store the result
    data = pca_encode_images("dataset/Mushrooms", model_weights_path="3_93_model.pth")
    save_json(data, "encoded.json")