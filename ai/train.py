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
import PIL.ImageOps

def train(
    model,
    train_dataloader,
    test_dataloader,
    num_epochs,
    optimizer,
    ):

    # create a folder to store model outputs
    try:
        new_folder_number = max([int(x) for x in os.listdir('results')]) + 1
    except:
        new_folder_number = 0
    new_folder_path = f'results/{new_folder_number}'
    os.mkdir(new_folder_path)

    # initialise the data logger
    data_logger = {
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
        'lr': [],
    }

    to_pil = T.ToPILImage()

    # how often to run the model on the test set
    eval_steps = len(train_dataloader) // 4

    scaler = torch.cuda.amp.GradScaler()

    # dymanic scheduler to reduce the learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=2,
        threshold=0.005,
        cooldown=1,
    )

    for epoch in tqdm(range(1, num_epochs+1), desc='Epoch', position=0):
        
        for iter, batch in enumerate(tqdm(train_dataloader, desc='Batch', position=1, leave=False)):
            model.train()

            with torch.cuda.amp.autocast():
                # forward pass
                batch = batch.float()
                batch = batch.to(model.device)
                pred, mu, log_var = model(batch)

                kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
                mse_loss = torch.nn.MSELoss()(batch, pred)
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

            # log data
            data_logger['epoch'].append(epoch)
            data_logger['train_loss'].append(loss.item())
            data_logger['test_loss'].append(test_loss)
            data_logger['lr'].append(scheduler.optimizer.param_groups[0]['lr'])

            # save data logs
            with open(f'{new_folder_path}/log.json', 'w') as fp:
                json.dump(data_logger, fp)

        scheduler.step(test_loss)

        # save model weights every epoch
        torch.save(model.state_dict(), f"{new_folder_path}/{epoch}_{iter}_model.pth")

def test(model, test_dataloader, epoch, iter, new_folder_path):
    to_pil = T.ToPILImage()
    total_test_loss = 0
    with torch.no_grad():
        with torch.cuda.amp.autocast():

            for batch in tqdm(test_dataloader, desc='Test', position=2, leave=False):
                model.eval()
                # forward pass calculate loss
                batch = batch.float()
                batch = batch.to(model.device)
                pred, mu, log_var = model(batch)
                test_loss = F.mse_loss(pred, batch)
                total_test_loss += test_loss.item()

            # save a sample from the test set
            sample_in, sample_out = batch[0], pred[0]

            sample_in, sample_out = to_pil(sample_in), to_pil(sample_out)
            # sample_in, sample_out = PIL.ImageOps.invert(sample_in), PIL.ImageOps.invert(sample_out)

            sample_in.save(f"{new_folder_path}/{epoch}_{iter}_sample_in.png")
            sample_out.save(f"{new_folder_path}/{epoch}_{iter}_sample_out.png")

    return total_test_loss / len(test_dataloader)


if __name__ == '__main__':

    # initialise argument parser
    parser = argparse.ArgumentParser(description='This file is for running training')

    # define arguments
    parser.add_argument('-d','--dataset_path', type=str, required=True,
                        help='The path to the directory containing the images, should have structure "dataset_name/class/*.jpg"')
    
    parser.add_argument('-b','--batch_size', type=int, default=64,
                        help='Batch size to use in the dataloaders')
    
    parser.add_argument('-w','--weights_path', type=str, required=False,
                        help='The path to pretrained weights')
    
    parser.add_argument('-e','--num_epochs', type=int, default=20,
                        help='The number of epochs to run training for, defaults to 20')
    
    parser.add_argument('-l','--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    
    parser.add_argument('-s','--split_ratio', type=float, default=0.9,
                        help='What fraction of data to use for training (rest is used for testing)')

    # extract arguments
    args = parser.parse_args()

    # get dataloaders
    train_dataloader, test_dataloader = get_dataloaders(dataset_path=args.dataset_path, batch_size=args.batch_size, split_ratio=args.split_ratio)

    # get model
    model = VAE(channel_in=3, latent_channels=512, ch=64)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.device = device

    if args.weights_path is not None:
        weights = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(weights)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
          num_epochs=args.num_epochs, optimizer=optimizer)
