#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import smdebug.pytorch as smd
import json
import os
from torchvision import datasets

#TODO: Import dependencies for Debugging and Profiling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction = "sum").item()
            pred = output.argmax(dim = 1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test(test_loader.dataset))
        )
     )      

def train(model, train_loader,validation_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    hook.set_mode(smd.modes.TRAIN)
    epochs = 10
    steps = 0
    running_loss = 0
    print_every = 5

    for e in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                correct = 0
                #validation
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps= model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        #calculate accuracy
                        pred = logps.argmax(dim = 1, keepdim = True)
                        correct += pred.eq(labels.view_as(pred)).sum().item()

                test_loss /= len(validation_loader.dataset)

                print(
                    "\nTest set : Average loss: {:.4f}, Accuracy: {}/{} ({:.0.f}%\n".format(
                        test_loss, correct, len(validation_loader.dataset), 100.0 * correct / len(validation_loader.dataset)
                    )
                )
                print(f"Epochs {epochs+1}/{epochs}..."
                f"Train loss: {running_loss/ print_every:.3f}..."
                f"validation loss: {test_loss / len(validation_loader):.3f}..."
                f"Validation Accuracy: {accuracy/len(validation_loader):.3f}")

                running_loss = 0
                model.train()

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    #Using densenet121
    model = models.densenet121(pretrained = True)   
    #Freeze parameters
    for param in model.parameters():
     param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2),
        nn.logSoftmax(dim = 1)
    )
    model = model.to(device)
    return model()


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    #transforms
    
    return torch.utils.data.DataLoader(data, batch_size, shuffle = True)



def main(args):
    #train_kwargs = {"batch_size": args.batch_size}
    #validate_kwargs = {"validation_batch_size": args.validation_batch}
    #test_kwargs = {"test_batch_size": args.test_batch_size}

#train_dir = "s3://sagemaker-us-east-1-535015539155/Data/train"
#validate_dir = "s3://sagemaker-us-east-1-535015539155/Data/validation"
#test_dir = "s3://sagemaker-us-east-1-535015539155/Data/test"


#Loading and Preprocessing of Data
    transforms = transforms.Compose([
            transforms.RandomResizeCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ])

    train_data = datasets.ImageFolder(args.data_dir + '/train', transform = transforms)
    test_data = datasets.ImageFolder(args.data_dir + '/test', transform = transforms)
    validation_data = datasets.ImageFolder(args.data_dir + '/validation', transform = transforms)



    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = args.validation_batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.test_batch_size, shuffle = True)


    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), "model_cnn.pt")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch-size",
        type = int,
        default= 64,
        metavar = "N",
        help = "input batch size for training (default: 64)"
     )
    parser.add_argument(
        "--validation-batch-size",
        type = int,
        default= 1000,
        metavar= "N",
        help = "input batch size for validation (default: 1000"
     )
    parser.add_argument(
        "--test-batch-size",
        type = int,
        default= 1000,
        metavar= "N",
        help = "input batch size for validation (default: 1000"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default= 14,
        metavar= "N",
        help = "number of epochs to train (default)"
    )
    parser.add_argument(
        "--lr",
        type = float,
        default=1.0,
        metavar= "LR",
        help = "learning rate (default: 1.0",
    )
    
    # Container environment
    parser.add_argument(
        "--hosts", 
        type=list, 
        default=json.loads(os.environ["SM_HOSTS"])
        )
    parser.add_argument(
        "--current-host", 
        type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument(
        "--num-gpus", 
        type=int, 
        default=os.environ["SM_NUM_GPUS"])
    args=parser.parse_args()
    
    main(args)
