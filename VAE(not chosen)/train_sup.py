from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import models
from data_utils import  *
from model import AutoEncoder


# Training settings
parser = argparse.ArgumentParser(description='STL10 SSL')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


train_dataset = get_train_label_dataset()
val_dataset = get_test_dataset()
train_loader = get_loader(train_dataset)
val_loader = get_loader(val_dataset)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script

n_classes = 10


class Net(nn.Module):
    def __init__(self, model, nclasses):
        super(Net, self).__init__()
        self.vae = model
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = self.vae(x)
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


    def train(self):
        super(Net, self).train()
        self.vae.train()

    def eval(self):
        super(Net, self).eval()
        self.vae.eval()


vae_model = AutoEncoder()

vae_model.load_state_dict(torch.load("./unsup_weights/best_train.pth"))

model = Net(vae_model, 10)


if torch.cuda.is_available():
    print("cuda available. \n")
    model = model.cuda()
else:
    print("cuda not available. \n")

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = nn.CrossEntropyLoss()

train_loss = []
val_loss = []

def train(epoch):
    model.train()
    avg_train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        avg_train_loss += loss.data.item() / len(train_loader)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data.item())
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    train_loss.append(avg_train_loss)

accus = []
best_accuracy = 0.0

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data), Variable(target)
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        validation_loss += criterion(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        accuracy))

    accus.append(accuracy)
    val_loss.append(validation_loss)
    return accuracy


for epoch in range(1, args.epochs + 1):
    train(epoch)
    accuracy = validation()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
    #model_file = 'model_' + str(epoch) + '.pth'
    #torch.save(model.state_dict(), model_file)
    #print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')

print('\nBest Accuracy: ({:.0f}%)\n'.format(accuracy))

np.savez_compressed("models/results", accus = np.array(accus), train_loss = np.array(train_loss),
                    val_loss = np.array(val_loss))


