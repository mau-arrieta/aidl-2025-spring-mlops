import torch
from torch.utils.data import random_split
import numpy as np
from dataset import MyDataset
from model import MyModel
from utils import accuracy

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
images_path = "/Users/mauarrieta/Desktop/Catalunya UPC docs/Labs MLops/Chinese MNIST data/data/data"
labels_path = "/Users/mauarrieta/Desktop/Catalunya UPC docs/Labs MLops/Chinese MNIST data/chinese_mnist.csv"


def custom_transform(image):
    # Convert to grayscale
    image = image.convert("L")  

    # Resize image (ensure consistent input size)
    image = image.resize((64, 64))  

    # Convert to NumPy array
    image = np.array(image, dtype=np.float32)  

    # Normalize (scale to [-1,1])
    image = (image / 255.0 - 0.5) * 2  

    # Add channel dimension (C, H, W) -> (1, 64, 64)
    image = np.expand_dims(image, axis=0)  

    # Convert to PyTorch tensor
    return torch.tensor(image, dtype=torch.float32)






def train_single_epoch():
    # TODO: Implement training loop
    raise NotImplementedError


def eval_single_epoch():
    # TODO: Implement evaluation loop
    raise NotImplementedError


def train_model(config):
    
    my_dataset = MyDataset()
    my_model = MyModel(...).to(device)

    for epoch in range(config["epochs"]):
        train_single_epoch(...)
        eval_single_epoch(...)

    return my_model


if __name__ == "__main__":
    my_dataset = MyDataset(images_path,labels_path,custom_transform)
    print(my_dataset.__getitem__(4))
    '''
    config = {
        "hyperparam_1": 1,
        "hyperparam_2": 2,
    }
    train_model(config)
    '''
    