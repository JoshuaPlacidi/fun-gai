from data import get_dataloaders
from model import VAE
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch
import torchvision.transforms as T
from tqdm import tqdm
import os
import json

def train(
    model,
    train_dataloader,
    test_dataloader,
    num_epochs,
    lr,
    optimizer,
    ):

    # create a folder to store model outputs
    try:
        new_folder_number = max([int(x) for x in os.listdir('results')]) + 1
    except:
        new_folder_number = 0
    new_folder_path = f'results/{new_folder_number}'
    os.mkdir(new_folder_path)

    data_logger = {
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
    }

    eval_steps = len(train_dataloader) // 4

    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm(range(1, num_epochs+1), desc='Epoch', position=0):
        
        for iter, batch in enumerate(tqdm(train_dataloader, desc='Batch', position=1, leave=False)):
            model.train()

            with torch.cuda.amp.autocast():
                # forward pass
                batch = batch.float()
                # TODO put the batch onto the same device as the model
                batch.to(model.device)
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

            if iter % eval_steps == 0:
                
                test_loss = test(model, test_dataloader, epoch, iter, new_folder_path)

            data_logger['epoch'].append(epoch)
            data_logger['train_loss'].append(loss)
            data_logger['test_loss'].append(test_loss)

            with open(f'results/{new_folder_path}/log.json', 'w') as fp:
                json.dump(data_logger, fp)

def test(model, test_dataloader, epoch, iter, new_folder_path):
    to_pil = T.ToPILImage()
    total_test_loss = 0
    with torch.no_grad():
        with torch.cuda.amp.autocast():

            for batch in tqdm(test_dataloader, desc='Test', position=2, leave=False):
                model.eval()
                # forward pass calculate loss
                batch = batch.float()
                pred, mu, log_var = model(batch.to(model.device))
                test_loss = F.mse_loss(pred, batch)
                total_test_loss += test_loss.item()

            # save a sample from the test set
            sample_in, sample_out = batch[0], pred[0]

            sample_in, sample_out = to_pil(sample_in), to_pil(sample_out)

            sample_in.save(f"{new_folder_path}/{epoch}_{iter}_sample_in.png")
            sample_out.save(f"{new_folder_path}/{epoch}_{iter}_sample_out.png")

    return total_test_loss / len(test_dataloader)


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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.device = device

    if args.weights_path is not None:
        model_dict = model.state_dict()
        weights = torch.load(args.weights_path, map_location=torch.device(args.cuda_device))
        model_dict.update(weights['state_dict'])
        # TODO fix model loading weights
        # model.load_state_dict(model_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
          num_epochs=args.num_epochs, lr=args.learning_rate, optimizer=optimizer)
