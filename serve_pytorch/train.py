import argparse
import datetime
import pathlib
import joblib
import json
import os
import pandas as pd

# torch imports
import torch
import torch.optim as optim
from torch.utils import tensorboard
import torch.utils.data
from torch import nn
from torch.utils.data.dataset import random_split

# imports the model in model.py by name
from model import VisionTransformer

# Progress bar
from tqdm import tqdm

# Monitor training
from torch.utils.tensorboard import SummaryWriter


def model_fn(MODEL_DIR):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    MODEL_INFO_PATH = MODEL_DIR / 'model_info.pth'
    MODEL_PATH = MODEL_DIR / 'model.pth'

    # First, load the parameters used to create the model.
    model_info = {}
    with open(MODEL_INFO_PATH, 'rb') as f:
        model_info = torch.load(f)
    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(image_size=model_info['image_size'], patch_size=model_info['patch_size'], 
                              num_classes=model_info['num_classes'], channels=model_info['channels'],
                               k=model_info['k'], depth=model_info['depth'], 
                               heads=model_info['heads'], mlp_dim=model_info['mlp_dim'])

    with open(MODEL_PATH, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

# Gets training data in batches from the train.csv file
def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    with open(os.path.join(training_dir, "train_X.z"), 'rb') as f:
        train_x = joblib.load(f)

    with open(os.path.join(training_dir, "train_Y.z"), 'rb') as f:
        train_y = joblib.load(f)

    train_y = torch.from_numpy(train_y).float().squeeze()
    train_x = torch.from_numpy(train_x).float()

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size), len(train_x)

    # Gets training data in batches from the train.csv file
def _get_train_and_validation_loader(batch_size, training_dir, split=0.1):
    print("Get train data and valid data loaders.")

    with open(os.path.join(training_dir, "train_X.z"), 'rb') as f:
        train_x = joblib.load(f)

    with open(os.path.join(training_dir, "train_Y.z"), 'rb') as f:
        train_y = joblib.load(f)

    train_y = torch.from_numpy(train_y).float().squeeze()
    train_x = torch.from_numpy(train_x).float()

    # find num elements in training and validation
    num_samples = train_x.shape[0]
    num_train = int(num_samples * split)
    num_valid = num_samples - num_train

    assert num_valid + num_train == num_samples

    # load all data and split into training and validation
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    train_ds, valid_ds = random_split(train_ds, lengths=[num_train, num_valid])

    # make dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    valid_dataloader =torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)

    return train_dataloader, valid_dataloader

# Provided training function
def train(model, train_loader, valid_loader, epochs, criterion, optimizer, device, tensorboard_monitor: bool = False):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    print('starting training.')

    # training loop is provided
    for epoch in range(1, epochs + 1):
        model.train() # Make sure that the model is in training mode.

        epoch_loss = 0
        batch_loss = 0

        model.train()
        for i, batch in enumerate(train_loader):

            # get data
            batch_x, batch_y = batch

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # get predictions from model
            y_pred = model(batch_x)
            
            # perform backprop
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            # get losses
            batch_loss += loss.data.item()
            epoch_loss += loss.data.item()

            # add batch loss to tensorboard
            if i % 100 == 0 and tensorboard_monitor:
                writer.add_scalar('batch loss', batch_loss / 100, epoch * len(train_loader) + i)
            batch_loss = 0

        # get validation
        model.eval()
        with torch.no_grad():

            # get valid loss
            epoch_valid_loss = 0

            for batch in valid_loader:
                
                # get data
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # get predictions on validation data
                y_pred = model(batch_x)

                # get loss
                loss = criterion(y_pred, batch_y)
                epoch_valid_loss += loss.data.item()
        
        print(y_pred)
        print(y_pred.shape)
        print(torch.nn.functional.sigmoid(y_pred))

        print("Epoch: {}, Loss: {}, Valid Loss: {}".format(epoch, epoch_loss / len(train_loader), epoch_valid_loss / len(valid_loader)))
        loss_dict = { 'epoch train loss': epoch_loss / len(train_loader),
                                                'epoch valid loss': epoch_valid_loss / len(valid_loader)}
        if tensorboard_monitor:
            writer.add_scalars('Loss', loss_dict, epoch)
    

if __name__ == '__main__':
    
    # sagemaker or local
    sagemaker_bool = False
    
    # retrieve base directory
    BASE_DIR = pathlib.Path().resolve()
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters or local; set automatically
    if sagemaker_bool:
        parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    else:
        DATA_DIR = BASE_DIR / 'data'
        MODEL_DIR = BASE_DIR / 'models'
        EXPORTS_DIR = DATA_DIR / 'exports'
        XRAY_LUNG_CLF_DIR = DATA_DIR / 'xray_lung_clf'
        EXPORTS_LUNGCLF_DIR = EXPORTS_DIR / 'xray_lung_clf'
        parser.add_argument('--output-data-dir', type=str, default=EXPORTS_LUNGCLF_DIR)
        parser.add_argument('--model-dir', type=str, default=MODEL_DIR)
        parser.add_argument('--data-dir', type=str, default=EXPORTS_LUNGCLF_DIR)

    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=84, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--image_size', type=int, default=256, metavar='N',
                        help='image_size (default: 256)')
    parser.add_argument('--patch_size', type=int, default=32, metavar='N',
                        help='patch_size (default: 32)')
    parser.add_argument('--num_classes', type=int, default=13, metavar='N',
                        help='num_classes (default: 13)')
    parser.add_argument('--channels', type=int, default=1, metavar='N',
                        help='channels (default: 1)')
    parser.add_argument('--k', type=int, default=64, metavar='N',
                        help='k (default: 64)')
    parser.add_argument('--depth', type=int, default=3, metavar='N',
                        help='depth (default: 3)')
    parser.add_argument('--heads', type=int, default=8, metavar='N',
                        help='heads (default: 8)')
    parser.add_argument('--mlp_dim', type=int, default=64, metavar='N',
                        help='mlp_dim (default: 64)')
    parser.add_argument('--valid', type=float, default=0.1, metavar='N',
                        help='fraction of training for validation (default: 10%)')
    parser.add_argument('--tensorboard', type=bool, default=True, metavar='N',
                        help='Add tensorboard monitor (default: True)')
    
    # args holds all passed-in arguments
    args = parser.parse_args()
    
    # save location as path object
    MODEL_DIR = pathlib.Path(args.model_dir).resolve()
    
    # save device and make replicable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))
    torch.manual_seed(args.seed)

    # Use tensorboard
    if args.tensorboard:
        print('creating tensorboard')
        RUNS_DIR = BASE_DIR / 'runs'
        LOGS_DIR = RUNS_DIR / 'logs/' / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        global writer, tensorboard_on
        writer = SummaryWriter(LOGS_DIR)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
    train_loader, valid_loader = _get_train_and_validation_loader(args.batch_size, args.data_dir, split = 0.1)

    # Build the model by passing in the input params
    model = VisionTransformer(image_size = args.image_size, patch_size=args.patch_size,                                                           num_classes=args.num_classes, 
                              channels = args.channels, k = args.k, depth = args.depth, heads = args.heads, 
                              mlp_dim = args.mlp_dim).to(device)

    # Get a batch for tensorboard
    if args.tensorboard:
        batch_X, batch_Y = next(iter(train_loader))
        print(f'X batch has shape {batch_X.shape}')
        writer.add_graph(model, batch_X)

    # Define optimizer and loss function for training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00008, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, valid_loader, args.epochs, criterion, optimizer, device, 
          tensorboard_monitor=args.tensorboard)

    # close tensorboard
    if args.tensorboard:
        writer.close()
        
    # Save model specifications
    MODEL_INFO_PATH = MODEL_DIR / 'model_info.pth'
        
    with open(MODEL_INFO_PATH, 'wb') as f:
        model_info = {
            'image_size': args.image_size,
            'patch_size': args.patch_size,
            'num_classes': args.num_classes,
            'channels': args.channels,
            'k': args.k,
            'depth': args.depth,
            'heads': args.heads,
            'mlp_dim': args.mlp_dim
        }
        torch.save(model_info, f)
    
    # Save the model parameters
    MODEL_PATH = MODEL_DIR / 'model.pth'
    
    with open(MODEL_PATH, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
        
    # Test model loading function
    model_fn(MODEL_DIR)
    