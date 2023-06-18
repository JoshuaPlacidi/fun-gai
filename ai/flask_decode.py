from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import torch
import numpy as np
from inference import decode, encode
import pickle as pk
import json
import math



app = Flask(__name__)

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

pca = pk.load(open('pca.pkl', 'rb'))
app.centroids = json.load(open('centroids.json', 'r'))


def similarity_list(new_coords, species_coords):

    distance_list = []
    for species_name, species_coords in species_coords.items():
            d = math.dist(new_coords, species_coords)

            distance_list.append((species_name, d))

    softmax_list = []
    for (name, dist) in distance_list:
          sf_dist = np.exp(dist) / sum([np.exp(x[1]) for x in distance_list])
          softmax_list.append((name, sf_dist))

    return sorted(softmax_list, key=lambda x: x[1])[::-1]



@app.route('/upload', methods=['POST'])
def upload_route():
    """
    Receives an image and returns the encoded latent vector
    """
    # Get the image from the request
    image = request.files['image'].read()

    # Get the latent representation of the image
    latent = encode(image, model_weights_path="3_93_model.pth")

    # Convert the latent representation to a numpy array and reshape it to 2D
    latent = np.array(latent)
    latent = latent.reshape(latent.shape[0], -1)

    # Apply PCA to the latent vector
    latent = app.pca.transform(latent)

    # Compute distance between encoded image and centroids
    distance_list = similarity_list(latent, app.centroids)

    # Return the encoded image as a json object
    return jsonify({'latent': latent.tolist()}), 200
    

@app.route('/decode', methods=['POST'])
@cross_origin()
def decode_route():

    data = request.get_json()['zSample']
    latents = torch.tensor(data)

    # Invert pca
    reconstructed_latents = pca.inverse_transform(latents)
    reconstructed_latents = torch.from_numpy(reconstructed_latents)
    reshaped_tensor = reconstructed_latents.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


    try:
        decoded_image = decode(reshaped_tensor, model_weights_path="3_93_model.pth")
        print(decoded_image.shape)
        # Convert the tensor to a list
        decoded_image_list = decoded_image.tolist()

        return jsonify({'decoded_image': decoded_image_list}), 200

    except Exception as e:
        return jsonify({'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)
