#!/usr/bin/env python
from PIL import Image
import os
import argparse
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import albumentations as alb

class AutoEncoderDataset(Dataset):
    def __init__(self, image_paths: list[str]):

        self.image_paths = image_paths

        # define image transformations
        self.transforms = transforms.Compose([
                    transforms.PILToTensor(),
                    transforms.Resize((64,64))
                ])
        self.alb = alb.Compose([
            alb.RandomBrightnessContrast(p=0.5),
            alb.HorizontalFlip(p=0.5),
            alb.Rotate(limit=30, p=0.5),
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def transform(self, image):
        # TODO add some image transformations here
        image = self.transforms(image) 
        # image = self.alb(image=image.numpy())['image']
        # image = torch.from_numpy(image)
        return image
    
    def __getitem__(self, index):
        # read the datapoint
        class_name, image_path = self.image_paths[index]
        
        #try:
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        #except:
            #raise Exception(f'Failed to read and process file: {image_path}')

        return image
    

def get_image_paths(dataset_path: str):
    # get the class folder names, removing hidden os folders
    class_folders = [f for f in os.listdir(dataset_path) if not f.startswith('.')]

    # initialise list for storing image paths
    image_list = []
    for class_name in class_folders:
        
        # get a list of the full paths of each image
        image_paths = [os.path.join(dataset_path, class_name, i) for i in os.listdir(os.path.join(dataset_path, class_name)) if not i.startswith('.')]
        
        # add the class name to the image path
        class_tuples = zip([class_name] * len(image_paths), image_paths)

        image_list += class_tuples

    return image_list    


def get_dataloaders(dataset_path: str, batch_size: int = 16, split_ratio: str = 0.9):
    import random

    # get a shuffled list of images
    image_paths = get_image_paths(dataset_path)
    random.shuffle(image_paths)

    # split into train and test
    split_idx = int(len(image_paths) * split_ratio)

    train_paths, test_paths = image_paths[:split_idx], image_paths[split_idx:]

    train_dataset, test_dateset = AutoEncoderDataset(train_paths), AutoEncoderDataset(test_paths)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dateset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

if __name__ == '__main__':

    # initialise argument parser
    parser = argparse.ArgumentParser(description='This file is for generating ML datasets')

    # set arguments
    parser.add_argument('-d','--dataset_path', type=str, required=True,
                        help='The path to the directory containing the images, should have structure "dataset_name/class/*.jpg"')

    # extract arguments
    args = parser.parse_args()

    image_paths = get_image_paths(dataset_path=args.dataset_path)
    dataset = AutoEncoderDataset(image_paths=image_paths)

    # visualise an image
    x = dataset.__getitem__(233)
    image = transforms.ToPILImage()(x)
    image.show()

    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    
    for batch in dataloader:
        print(batch.shape)
        break
