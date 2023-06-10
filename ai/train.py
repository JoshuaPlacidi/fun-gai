from data import get_dataloaders
from model import VAE
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch

def train(
    model,
    train_dataloader,
    test_dataloader,
    num_epochs,
    lr,
    optimizer,
    ):
    # TODO train model here and save its best weights

    data_logger = {
        'epoch': [],
        'loss': [],

    }

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, num_epochs+1):
        
        for batch in train_dataloader:
            # forward pass
            batch = batch.float()
            # TODO put the batch onto the same device as the model
            # batch.to(model)
            pred, mu, log_var = model(batch)

            # calculate loss
            # TODO add the KL divergence
            kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
            mse_loss = F.mse_loss(pred, batch)
            loss = kl_loss + mse_loss

            # back pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
            scaler.step(optimizer)
            scaler.update()

            data_logger['epoch'].append(epoch)
            data_logger['loss'].append(loss)

    return

if __name__ == '__main__':

    # initialise argument parser
    parser = argparse.ArgumentParser(description='This file is for running training')

    # define arguments
    parser.add_argument('-d','--dataset_path', type=str, required=True,
                        help='The path to the directory containing the images, should have structure "dataset_name/class/*.jpg"')
    
    parser.add_argument('-b','--batch_size', type=str, default=64,
                        help='Batch size to use in the dataloaders')
    
    parser.add_argument('-w','--weights_path', type=str, required=False,
                        help='The path to pretrained weights')
    
    parser.add_argument('-c','--cuda_device', type=str, default='cpu',
                        help='The cuda device to run training with, e.g. "cuda:0", defaults to "cpu"')
    
    parser.add_argument('-e','--num_epochs', type=int, default=20,
                        help='The number of epochs to run training for, defaults to 20')
    
    parser.add_argument('-l','--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')

    # extract arguments
    args = parser.parse_args()

    # get dataloaders
    train_dataloader, test_dataloader = get_dataloaders(dataset_path=args.dataset_path, batch_size=args.batch_size)

    # get model
    model = VAE(channel_in=3, latent_channels=64)

    if args.cuda_device is not None:
        model.to(args.cuda_device)

    if args.weights_path is not None:
        model_dict = model.state_dict()
        weights = torch.load(args.weights_path, map_location=torch.device(args.cuda_device))
        model_dict.update(weights['state_dict'])
        # TODO fix model loading weights
        # model.load_state_dict(model_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
          num_epochs=args.num_epochs, lr=args.learning_rate, optimizer=optimizer)
