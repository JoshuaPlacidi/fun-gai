# Fun-GAI: hallucinating mushrooms with generative AI

TODO change to have 3 different gifs
<p align="center">
  <img width="250" height="250" src="ai/fungai.gif">
  <img width="250" height="250" src="ai/fungai.gif">
  <img width="250" height="250" src="ai/fungai.gif">
</p>

Authors: [Joshua Placidi](https://www.linkedin.com/in/joshua-placidi/), [Sara Sabzikari](https://www.linkedin.com/in/sara-sabzikari/), [Vincenzo Incutti](https://www.linkedin.com/in/vincenzo-incutti/), [Ka Yeon Kim](https://www.linkedin.com/in/ka-yeon-kim-298935216/)

### Introduction
This is a project originally built in 12 hours for a [Biology + Generative Artifical Intelligence Hackathon](https://biohacklondon.notion.site/BioHack-London-40bea186f1a24e779b276087f2ee7e61).
We trained a [variational auto-encoder](https://en.wikipedia.org/wiki/Variational_autoencoder) to learn a latent space represenation of the physiology of mushrooms, the fruiting body of fungi.


It has been estimated that more than 90% of all fungal species have yet to be described by science ![1](https://www.bbc.co.uk/news/science-environment-64251382).
We built and trained a VAE from scratch to synthesis what new, yet undiscovered, mushrooms *could* look like.
This project was built as a fun exploration into how an auto-encoder model learns to represent images of mushrooms in 
a latent space.
The culmination of our work can be seen in the gif at the top of the page.

### Biological

### Technical

VAEs learn in a self-supervised manner to predict their own input, given an input $X^{3,224,224}$ the model produces an output $\hat{X}^{3,224,224}$ with the objective of minimising the difference between $X$ and $\hat{X}$.
The model has an encoder-decoder structure with a bottle neck in the middle, the bottle neck forces the encoder to learn to compress the input into a latent representation $z = encoder(X)$ which is then given to the decoder to try and project back to original input $\hat{X} = decoder(z)$.
The idea is that the VAE has to learn extract useful information to store in $z$.
We train the model to minimize the reconstruction loss which is measured as the [mean-squared error](https://en.wikipedia.org/wiki/Mean_squared_error) between $X$ and $\hat{X}$.
Additionally VAEs add a KL-divergence term to the loss function, encouraging the model to learn normally distributed latents, thus giving a more coherent latent space and enabling generative sampling.

To generate new samples using our learned model we simply pass a randomly initialised latent $z \sim \mathcal{N}(\mathbf{0}^{512},\mathbf{1}^{512})$ to the decoder: $Y = decoder(z)$.
