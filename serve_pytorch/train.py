import argparse
import pathlib
import joblib
import json
import os
import pandas as pd

# torch imports
import torch
import torch.optim as optim
import torch.utils.data
from torch import nn

# imports the model in model.py by name
from model import VisionTransformer

# Progress bar
from tqdm import tqdm

# Monitor training
from torch.utils.tensorboard import SummaryWriter

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    MODEL_INFO_PATH = MODEL_DIR / 'model_info.pth'
    MODEL_PATH = MODEL_DIR / 'model.pth'

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
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

# Provided training function
def train(model, train_loader, epochs, criterion, optimizer, device):
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
    
    # training loop is provided
    for epoch in range(1, epochs + 1):
        model.train() # Make sure that the model is in training mode.

        epoch_loss = 0
        batch_loss = 0

        for i, batch in tqdm(enumerate(train_loader)):

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
            if i % 100 == 0:
                writer.add_scalar('batch loss', batch_loss / 100, epoch * len(train_loader) + i)
                batch_loss = 0

        print("Epoch: {}, Loss: {}".format(epoch, epoch_loss / len(train_loader)))
        writer.add_scalar('epoch loss', epoch_loss / len(train_loader), epoch)


if __name__ == '__main__':

    BASE_DIR = pathlib.Path().resolve().parent

    DATA_DIR = BASE_DIR / 'data'
    MODEL_DIR = BASE_DIR / 'models'
    RUNS_DIR = BASE_DIR / 'runs'

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    EXPORTS_DIR = DATA_DIR / 'exports'
    XRAY_LUNG_CLF_DIR = DATA_DIR / 'xray_lung_clf'
    EXPORTS_LUNGCLF_DIR = EXPORTS_DIR / 'xray_lung_clf'
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    #parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    #parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    #parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--output-data-dir', type=str, default=EXPORTS_LUNGCLF_DIR)
    parser.add_argument('--model-dir', type=str, default=MODEL_DIR)
    parser.add_argument('--data-dir', type=str, default=EXPORTS_LUNGCLF_DIR)
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
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
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Use tensorboard
    print('creating tensorboard')
    global writer
    writer = SummaryWriter(RUNS_DIR)

    # Load the training data.
    train_loader, num_samples = _get_train_data_loader(args.batch_size, args.data_dir)
    train_loader = _get_train_and_validation_loader(train_loader, num_samples=num_samples, fraction=0.1)

    # Build the model by passing in the input params
    # To get params from the parser, call args.argument_name, ex. args.epochs or ards.hidden_dim
    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    model = VisionTransformer(image_size = args.image_size, patch_size = args.patch_size, num_classes = args.num_classes, 
                              channels = args.channels, k = args.k, depth = args.depth, heads = args.heads, mlp_dim = args.mlp_dim)

    # Get a batch for tensorboard
    batch_X, batch_Y = next(iter(train_loader))
    print(f'X batch has shape {batch_X.shape}')
    writer.add_graph(model, batch_X)

    ## Define optimizer and loss function for training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00008, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # Trains the model (given line of code, which calls the above training function)
    #train(model, train_loader, args.epochs, criterion, optimizer, device)
    writer.close()

    # Keep the keys of this dictionary as they are 
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

    # Test model loading functiogn
    model_fn(MODEL_DIR)