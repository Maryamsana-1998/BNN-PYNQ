import argparse
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from .binarized_modules import *
from .dataloaders.aider import AIDER, aider_transforms, squeeze_transforms

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Quantized Inception (MNIST) Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--root-dir', type=str, default='.', help='path to the root dir of AIDER')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--gpus', default=0, help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default=False, action='store_true', help='Perform only evaluation on val dataset.')
parser.add_argument('--wb', type=int, default=1, metavar='N', choices=[1, 2, 4],
                    help='number of bits for weights (default: 1)')
parser.add_argument('--ab', type=int, default=1, metavar='N', choices=[1, 2, 4],
                    help='number of bits for activations (default: 1)')
parser.add_argument('--eval', default=False, action='store_true', help='perform evaluation of trained model')
parser.add_argument('--export', default=False, action='store_true',
                    help='perform weights export as npz of trained model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
save_path = 'results/ernet-w{}a{}.pt'.format(args.wb, args.ab)
prev_acc = 0
debug = False


def init_weights(m):
    if type(m) == BinarizeLinear or type(m) == BinarizeConv2d:
        torch.nn.init.uniform_(m.weight, -1, 1)
        m.bias.data.fill_(0.01)


class BinarizeConv2d_block(nn.Module):
    def __init__(self, wb, ifm_ch, num_filt, kernel_size, stride, padding, bias):
        super(BinarizeConv2d_block, self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(wb, ifm_ch, num_filt, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_filt),
            nn.Hardtanh(inplace=True),
            Quantizer(args.ab))

    def forward(self, x):
        return self.features(x)


class ACFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        ''' 
        Dilated Convolution

        i = input
        o = output
        p = padding
        k = kernel_size
        s = stride
        d = dilation
        
        o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        '''

        self.conv1 = BinarizeConv2d(args.wb, in_channels, in_channels, kernel_size=3, stride=1, padding=0, dilation=1,
                                    groups=in_channels, bias=True)
        self.conv2 = BinarizeConv2d(args.wb, in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=2,
                                    groups=in_channels, bias=True)
        self.conv3 = BinarizeConv2d(args.wb, in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=3,
                                    groups=in_channels, bias=True)
        self.fused_conv = BinarizeConv2d(args.wb, in_channels * 3, out_channels, kernel_size=1, stride=1, padding=0,
                                         dilation=1,
                                         bias=True)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        if debug:
            print('Shape of input in ACFF Forward= ', x.shape)
            print('Output of layer1(x): ', self.conv1(x).shape)
            print('Output of layer2(x): ', self.conv2(x).shape)
            print('Output of layer3(x): ', self.conv3(x).shape)

        # Fusion
        out = torch.cat((self.conv1(x), self.conv2(x), self.conv3(x)), 1)

        if debug:
            print('Shape after concat in ACFF forward: ', out.shape)

        out = self.fused_conv(out)
        out = self.leaky_relu(out)
        out = self.batch_norm(out)
        out = self.dropout(out)

        if debug:
            print('Final shape of ACFF out: ', out.shape, '\n')

        return out


class ErNET(nn.Module):
    def __init__(self):
        super(ErNET, self).__init__()

        self.conv1 = BinarizeConv2d(args.wb, in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=0,
                                    bias=True)
        self.acff1 = ACFF(16, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acff2 = ACFF(64, 96)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acff3 = ACFF(96, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acff4 = ACFF(128, 128)
        self.acff5 = ACFF(128, 128)
        self.acff6 = ACFF(128, 256)
        self.conv2 = BinarizeConv2d(args.wb, in_channels=256, out_channels=5, kernel_size=1, stride=1, padding=0,
                                    bias=True)
        self.globalpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=0)
        self.fc = BinarizeLinear(args.wb, 3 * 3 * 5, 5, bias=True)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.acff1(out)
        out = self.pool1(out)
        out = self.acff2(out)
        out = self.pool2(out)
        out = self.acff3(out)
        out = self.pool3(out)
        out = self.acff4(out)
        out = self.acff5(out)
        out = self.acff6(out)
        out = self.conv2(out)
        out = self.globalpool(out)

        if debug:
            print('Shape of globalpool output: ', out.shape)

        out = out.view(-1, 5 * 3 * 3)
        out = self.fc(out)
        out = self.soft(out)

        if debug:
            print('Final shape of ErNET Output: ', out.shape)

        return out

    def export(self):
        import numpy as np
        dic = {}
        i = 0

        # process conv and BN layers
        for k in range(len(self.features)):
            if hasattr(self.features[k], 'weight') and not hasattr(self.features[k], 'running_mean'):
                dic['arr_' + str(i)] = self.features[k].weight.detach().numpy()
                i = i + 1
                dic['arr_' + str(i)] = self.features[k].bias.detach().numpy()
                i = i + 1
            elif hasattr(self.features[k], 'running_mean'):
                dic['arr_' + str(i)] = self.features[k].bias.detach().numpy()
                i = i + 1
                dic['arr_' + str(i)] = self.features[k].weight.detach().numpy()
                i = i + 1
                dic['arr_' + str(i)] = self.features[k].running_mean.detach().numpy()
                i = i + 1
                dic['arr_' + str(i)] = 1. / np.sqrt(self.features[k].running_var.detach().numpy())
                i = i + 1
            # process inception block
            if hasattr(self.features[k], 'conv1x1'):
                for j in range(len(self.features[k].conv1x1.features)):
                    if hasattr(self.features[k].conv1x1.features[j], 'weight') and not hasattr(
                            self.features[k].conv1x1.features[j], 'running_mean'):
                        dic['arr_' + str(i)] = self.features[k].conv1x1.features[j].weight.detach().numpy()
                        i = i + 1
                        dic['arr_' + str(i)] = self.features[k].conv1x1.features[j].bias.detach().numpy()
                        i = i + 1
                    elif hasattr(self.features[k].conv1x1.features[j], 'running_mean'):
                        dic['arr_' + str(i)] = self.features[k].conv1x1.features[j].bias.detach().numpy()
                        i = i + 1
                        dic['arr_' + str(i)] = self.features[k].conv1x1.features[j].weight.detach().numpy()
                        i = i + 1
                        dic['arr_' + str(i)] = self.features[k].conv1x1.features[j].running_mean.detach().numpy()
                        i = i + 1
                        dic['arr_' + str(i)] = 1. / np.sqrt(
                            self.features[k].conv1x1.features[j].running_var.detach().numpy())
                        i = i + 1

            if hasattr(self.features[k], 'conv3x3'):
                for j in range(len(self.features[k].conv3x3.features)):
                    if hasattr(self.features[k].conv3x3.features[j], 'weight') and not hasattr(
                            self.features[k].conv3x3.features[j], 'running_mean'):
                        dic['arr_' + str(i)] = self.features[k].conv3x3.features[j].weight.detach().numpy()
                        i = i + 1
                        dic['arr_' + str(i)] = self.features[k].conv3x3.features[j].bias.detach().numpy()
                        i = i + 1
                    elif hasattr(self.features[k].conv3x3.features[j], 'running_mean'):
                        dic['arr_' + str(i)] = self.features[k].conv3x3.features[j].bias.detach().numpy()
                        i = i + 1
                        dic['arr_' + str(i)] = self.features[k].conv3x3.features[j].weight.detach().numpy()
                        i = i + 1
                        dic['arr_' + str(i)] = self.features[k].conv3x3.features[j].running_mean.detach().numpy()
                        i = i + 1
                        dic['arr_' + str(i)] = 1. / np.sqrt(
                            self.features[k].conv3x3.features[j].running_var.detach().numpy())
                        i = i + 1

            if hasattr(self.features[k], 'conv5x5'):
                for j in range(len(self.features[k].conv5x5.features)):
                    if hasattr(self.features[k].conv5x5.features[j], 'weight') and not hasattr(
                            self.features[k].conv5x5.features[j], 'running_mean'):
                        dic['arr_' + str(i)] = self.features[k].conv5x5.features[j].weight.detach().numpy()
                        i = i + 1
                        dic['arr_' + str(i)] = self.features[k].conv5x5.features[j].bias.detach().numpy()
                        i = i + 1
                    elif hasattr(self.features[k].conv5x5.features[j], 'running_mean'):
                        dic['arr_' + str(i)] = self.features[k].conv5x5.features[j].bias.detach().numpy()
                        i = i + 1
                        dic['arr_' + str(i)] = self.features[k].conv5x5.features[j].weight.detach().numpy()
                        i = i + 1
                        dic['arr_' + str(i)] = self.features[k].conv5x5.features[j].running_mean.detach().numpy()
                        i = i + 1
                        dic['arr_' + str(i)] = 1. / np.sqrt(
                            self.features[k].conv5x5.features[j].running_var.detach().numpy())
                        i = i + 1

        # process linear and BN layers
        for k in range(len(self.classifier)):
            if hasattr(self.classifier[k], 'weight') and not hasattr(self.classifier[k], 'running_mean'):
                dic['arr_' + str(i)] = np.transpose(self.classifier[k].weight.detach().numpy())
                i = i + 1
                dic['arr_' + str(i)] = self.classifier[k].bias.detach().numpy()
                i = i + 1
            elif hasattr(self.classifier[k], 'running_mean'):
                dic['arr_' + str(i)] = self.classifier[k].bias.detach().numpy()
                i = i + 1
                dic['arr_' + str(i)] = self.classifier[k].weight.detach().numpy()
                i = i + 1
                dic['arr_' + str(i)] = self.classifier[k].running_mean.detach().numpy()
                i = i + 1
                dic['arr_' + str(i)] = 1. / np.sqrt(self.classifier[k].running_var.detach().numpy())
                i = i + 1
        if not os.path.isdir('results'):
            os.mkdir('results')

        save_file = 'results/ernet-w{}a{}.npz'.format(args.wb, args.ab)
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
            if hasattr(p, 'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.org.copy_(p.data.clamp_(-1, 1))
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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                                 len(test_loader.dataset), new_acc))
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
    model_transforms = aider_transforms
    transformed_dataset = AIDER("dataloaders/aider_labels.csv", args.root_dir, transform=model_transforms)

    train_set, test_set = torch.utils.data.random_split(transformed_dataset, [5432, 1000])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             drop_last=True)

    model = ErNET()
    if not os.path.isdir('results'):
        os.mkdir('results')

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
        model = torch.load(save_path, map_location='cpu')
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
            if epoch % 40 == 0:
                optimizer.param_groups[0]['lr'] = optimizer

    model = torch.load(save_path, map_location='cpu')
    #model.export().param_groups[0]['lr'] * 0.1
