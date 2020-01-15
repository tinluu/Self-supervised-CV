import argparse
import os
import copy
import time
import numpy as np
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.models as models
import torch.optim as optim
from model import *
from data_utils import *


parser = argparse.ArgumentParser(description='STL10 SSL')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


def unsup_loss(recon_x, x, mu, logvar):
    #print(recon_x.size())
    #print(x.view(-1, 3*96*96).size())
    
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3*96*96), reduction = 'sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch, model, data_loader, criterion, optimizer, log_interval=args.log_interval, device=device):
    start = time.time()
    model.train()
    train_loss = 0
    for i, (img, _) in enumerate(data_loader):
        img = img.to(device)
        recon_x, mu, logvar = model(img)
        optimizer.zero_grad()
        loss = criterion(recon_x, img.view(-1, 3*96*96), mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i % log_interval == 0:
            crt_time = time.time()
            print("Epoch {}: {} / {}: {:.2f}% ==> Running for {:.2f} seconds. Batch loss = {}.".format(epoch, i, len(data_loader), float(100.*i / len(data_loader)), crt_time-start, loss.item()))

    end = time.time()
    time_elapsed = end - start;
    print('Epoch {}: Training complete in {:.0f}m {:.0f}s'.format(epoch, time_elapsed//60, time_elapsed%60))
    print('Training loss = {}'.format(train_loss))
    return train_loss



def validation(epoch, model, data_loader, criterion, log_interval=args.log_interval, device=device):
    model.eval()
    start = time.time()
    with torch.no_grad():
        val_loss = 0
        for i, (img, _) in enumerate(data_loader):
            img = img.to(device)
            recon_x, mu, logvar = model(img)
            loss = criterion(recon_x, img.view(-1, 3*96*96), mu, logvar)
            val_loss += loss.item()
        end = time.time()
        time_elapsed = end - start;
        print('Epoch {}: Validation complete in {:.0f}m {:.0f}s'.format(epoch, time_elapsed//60, time_elapsed%60))
        print('Validation loss = {}'.format(val_loss))
        return val_loss


if __name__ == "__main__":
    
    save_path = './unsup_weights/best_train.pth'
    check_path = '/unsup_weights/autoen_encoder2_check_l2.pth'
    model = AutoEncoder()
    model = model.to(device)
    
    criterion = unsup_loss

    #optimizer_ft = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)

    num_epochs = args.epochs

    #best_train_loss = 1.e10
    best_val_loss = 1.e10

    unlabeled = get_dataset_all()
    train_sampler, val_sampler = train_val_sampler(unlabeled)
    unlabeled_train_loader = get_loader(unlabeled, batch_size=128, sampler=train_sampler)
    unlabeled_val_loader = get_loader(unlabeled, batch_size=128, sampler=val_sampler)


    for epoch in range(num_epochs):
        train_loss = train(epoch, model, data_loader=unlabeled_train_loader, criterion=criterion, optimizer=optimizer)
        # if train_loss < best_train_loss:
        #     torch.save(model_best.state_dict(), save_path)
        val_loss = validation(epoch, model, data_loader=unlabeled_val_loader, criterion=criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
        print(f'Best Validation loss is = {best_val_loss}')