from data import get_dataloaders
from model import ResNetAutoEncoder, get_configs
import argparse
import torch

def train(
    model,
    train_dataloader,
    test_dataloader,
    num_epochs,
    lr,
    ):
    # TODO train model here and save its best weights

    return

if __name__ == '__main__':

    # initialise argument parser
    parser = argparse.ArgumentParser(description='This file is for running training')

    # define arguments
    parser.add_argument('-d','--dataset_path', type=str, required=True,
                        help='The path to the directory containing the images, should have structure "dataset_name/class/*.jpg"')
    
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
    train_dataloader, test_dataloader = get_dataloaders(dataset_path=args.dataset_path, batch_size=1000)

    # get model
    configs, bottleneck = get_configs('resnet18')
    model = ResNetAutoEncoder(configs, bottleneck)

    if args.weights_path is not None:
        model_dict = model.state_dict()
        weights = torch.load(args.weights_path, map_location=torch.device(args.cuda_device))
        model_dict.update(weights['state_dict'])
        # TODO fix model loading weights
        # model.load_state_dict(model_dict)

    train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
          num_epochs=args.num_epochs, lr=args.learning_rate)
