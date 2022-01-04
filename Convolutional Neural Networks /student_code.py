# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms
import numpy as np


class LeNet(nn.Module):
    """
    
    1)One convolutional layer with the number of output channels to be 6, kernel size to be 5, stride to be 1, followed by a relu 
    activation layer and then a 2D max pooling layer (kernel size to be 2 and stride to be 2).
    2)One convolutional layer with the number of output channels to be 16, kernel size to be 5, stride to be 1, followed by a relu 
    activation layer and then a 2D max pooling layer (kernel size to be 2 and stride to be 2).
    3)A Flatten layer to convert the 3D tensor to a 1D tensor.
    4)A Linear layer with output dimension to be 256, followed by a relu activation function.
    5)A Linear layer with output dimension to be 128, followed by a relu activation function.
    6)A Linear layer with output dimension to be the number of classes (in our case, 100).
    
    Shape_dict should have the following form: {1: [a,b,c,d], 2:[e,f,g,h], .., 6: [x,y]}

    The linear layer and the convolutional layer have bias terms.
    """

    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2,  stride=2)
        
        self.conv_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)        
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2,  stride=2)
        
        self.fc1 = nn.Linear(16*5*5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        
    def forward(self, x):
        x.to('cuda')
        shape_dict = {}
        # certain operations
        #layer 1 as per assignment description
        x = nn.functional.relu(self.conv_1(x))
        x = self.max_pool_1(x)
        shape_dict[1] = list(x.shape)
        
        #layer 2 as per assignment description
        x = nn.functional.relu(self.conv_2(x))
        x = self.max_pool_2(x)
        shape_dict[2] = list(x.shape)
        
        #layer 3 (flatten) as per assignment description
        x = x.view(-1, 16*5*5)
        shape_dict[3] = list(x.shape)
        
        #layer 4 (FC1) as per assignment description
        x = nn.functional.relu(self.fc1(x))
        shape_dict[4] = list(x.shape)
        
        #layer 5 (FC2) as per assignment description
        x = nn.functional.relu(self.fc2(x))
        shape_dict[5] = list(x.shape)
        
        #layer 6 (FC3) as per assignment description
        out = self.fc3(x)
        shape_dict[6] = list(out.shape)
        
        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0
    for name, param in model.named_parameters():
        model_params += np.prod(list(param.size()))
    
    #The function output is in the unit of Million(1e6).
    model_params /= (1e6)
    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.to('cuda')
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        input, target = input.to('cuda'), target.to('cuda')  
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    #model.to('cuda')
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to('cuda'), target.to('cuda') 
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
