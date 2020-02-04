from __future__ import print_function
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules import *
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Quantized LeNet (MNIST) Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--gpus', default=0, help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--resume', default=False, action='store_true', help='Perform only evaluation on val dataset.')
parser.add_argument('--wb', type=int, default=1, metavar='N', choices=[1, 2, 4], help='number of bits for weights (default: 1)')
parser.add_argument('--ab', type=int, default=1, metavar='N', choices=[1, 2, 4], help='number of bits for activations (default: 1)')
parser.add_argument('--eval', default=False, action='store_true', help='perform evaluation of trained model')
parser.add_argument('--export', default=False, action='store_true', help='perform weights export as npz of trained model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
save_path='results/lenet-w{}a{}.pt'.format(args.wb, args.ab)
prev_acc = 0

def init_weights(m):
    if type(m) == BinarizeLinear or type(m) == BinarizeConv2d:
        torch.nn.init.uniform_(m.weight, -1, 1)
        m.bias.data.fill_(0.01)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(args.wb, 1, 1, kernel_size=3, stride=1, padding=0, bias=True,groups=1),  #depthwise convolution
            BinarizeConv2d(args.wb, 1, 32, kernel_size=1, stride=1, padding=0, bias=True),          #pointwise convolution
            nn.BatchNorm2d(32),
            nn.Hardtanh(inplace=True),
            Quantizer(args.ab),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinarizeConv2d(args.wb, 32, 32, kernel_size=5, padding=2, bias=True,groups=32),
            BinarizeConv2d(args.wb, 32, 64, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),
            Quantizer(args.ab),
        	nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            BinarizeLinear(args.wb, 6*6*64, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            Quantizer(args.ab),
            
            nn.Dropout(0.5),
            
            BinarizeLinear(args.wb, 1024, 10, bias=True),
            nn.BatchNorm1d(10),
            nn.LogSoftmax())

        self.features.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 6*6*64)
        x = self.classifier(x)
        return x
    
    def export(self):
        import numpy as np
        dic = {}
        i = 0
        
        # process conv and BN layers
        for k in range(len(self.features)):
            if hasattr(self.features[k], 'weight') and not hasattr(self.features[k], 'running_mean'):
                dic['arr_'+str(i)] = self.features[k].weight.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.features[k].bias.detach().numpy()
                i = i + 1
            elif hasattr(self.features[k], 'running_mean'):
                dic['arr_'+str(i)] = self.features[k].bias.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.features[k].weight.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.features[k].running_mean.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = 1./np.sqrt(self.features[k].running_var.detach().numpy())
                i = i + 1
        
        # process linear and BN layers
        for k in range(len(self.classifier)):
            if hasattr(self.classifier[k], 'weight') and not hasattr(self.classifier[k], 'running_mean'):
                dic['arr_'+str(i)] = np.transpose(self.classifier[k].weight.detach().numpy())
                i = i + 1
                dic['arr_'+str(i)] = self.classifier[k].bias.detach().numpy()
                i = i + 1
            elif hasattr(self.classifier[k], 'running_mean'):
                dic['arr_'+str(i)] = self.classifier[k].bias.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.classifier[k].weight.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.classifier[k].running_mean.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = 1./np.sqrt(self.classifier[k].running_var.detach().numpy())
                i = i + 1
        
        save_file = 'results/lenet-w{}a{}.npz'.format(args.wb, args.ab)
        np.savez(save_file, **dic)
        print("Model exported at: ", save_file)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def test(save_model=False):
    model.eval()
    test_loss = 0
    correct = 0
    global prev_acc
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += criterion(output, target).data
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    new_acc = 100. * correct.float() / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset), new_acc))
    if new_acc > prev_acc:
        # save model
        if save_model:
            torch.save(model, save_path)
            print("Model saved at: ", save_path, "\n")
        prev_acc = new_acc

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    if args.cuda:
    		torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    	batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    	batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = LeNet()
    if args.cuda:
    		torch.cuda.set_device(0)
    		model.cuda()
    criterion = nn.CrossEntropyLoss()
    # test model
    if args.eval:
        model = torch.load(save_path)
        test()
    # export npz
    elif args.export:
        model = torch.load(save_path, map_location = 'cpu')
        model.export()
    # train model
    else:
        if args.resume:
            model = torch.load(save_path)
            test()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)  
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(save_model=True)
            if epoch%40==0:
                optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1