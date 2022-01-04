# Abhay Kumar (kumar95)
# CS540: HW6

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    '''
    Input: an optional boolean argument (default value is True for training dataset)
    Return: Dataloader for the training set (if training = True) or the test set (if training = False)
    '''
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if training == True:
        train_set = datasets.MNIST('./data', train=True, download=True,
                       transform=custom_transform)
        data_set = train_set
    else:
        test_set = datasets.MNIST('./data', train=False,
                       transform=custom_transform)
        data_set = test_set
    loader = torch.utils.data.DataLoader(data_set, batch_size = 50)
    return loader



def build_model():
    '''
    Input: none.
    Return: an untrained neural network model
    '''  
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64,10)
        )
    
    return model




def train_model(model, train_loader, criterion, T):
    '''
    Input: the model produced by the previous function, the train DataLoader produced by the first function, the criterion, and the number of epochs T for training.
    Return: none
    '''
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    dataset_size = len(train_loader)
    for epoch in range(T):  # loop over the dataset multiple times
        num_correct = 0
        total = 0
        cum_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            
            cum_loss += loss.item()
            correct_count = 0
            pred = torch.argmax(outputs, 1)
            total += labels.size(0)
            num_correct += (pred == labels).sum().item()

        # print statistics
        print('Train Epoch: {} Accuracy: {}/{}({:.2f}%) Loss: {:.3f}'.format(epoch, num_correct, total , num_correct/total*100, cum_loss/total ))
    

    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 
    RETURNS:
        None
    """
    model.eval()
    num_correct = 0
    total = 0
    cum_loss = 0.0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            pred = torch.argmax(outputs, 1)
            
            cum_loss += loss.item()
            total += labels.size(0)
            num_correct += (pred == labels).sum().item()
            
    if show_loss == False:
        print('Accuracy: {:.2f}%'.format(num_correct/total*100))
    else:
        print('Average loss: {:.4f}'.format(cum_loss/total))
        print('Accuracy: {:.2f}%'.format(num_correct/total*100))
    
    


def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1
    RETURNS:
        None
    """
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    model.eval()
    
    img = test_images[index].view(1, 784) 
    
    with torch.no_grad():
        logits = model(img)
        prob = F.softmax(logits, dim=1)
    prob_sorted, indices = prob.sort(descending=True) 
    
    prob_sorted = prob_sorted.tolist()[0]
    indices = indices.tolist()[0]
    for i in range(3):
        print('{}: {:.2f}%'.format(class_names[indices[i]], prob_sorted[i]*100 ))
        

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    print("TEST: Data Loader\n\n")
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    print("TEST: Build Model\n\n", model)
    train_model(model, train_loader, criterion, T=5)
    print("\n\nTEST: Eval with show_loss=False\n\n")
    evaluate_model(model, test_loader, criterion, show_loss = False)  
    print("\n\nTEST: Eval with show_loss=True\n\n")
    evaluate_model(model, test_loader, criterion, show_loss = True)
    print("\n\nTEST: Individual Image Testing\n\n")
    pred_set_img_label =  next(iter(test_loader))
    pred_set_label = pred_set_img_label[1]
    pred_set = pred_set_img_label[0]
    index = 48
    predict_label(model, pred_set, index)
    print("\n GROUND TRUTH: ", pred_set_label[index])

