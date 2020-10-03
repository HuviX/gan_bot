import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from net import Discriminator, Generator

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=False, help='path to dataset')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nepoch', type=int, default=20, help='num of epochs')
parser.add_argument('--name', type = str, help = 'net name')
parser.add_argument('--outf', type = str, default='myoutf', help = 'net name')
opt = parser.parse_args()

if __name__ == "__main__":
    dataset = dset.ImageFolder(root=opt.data,
                                transform=transforms.Compose([
                                    transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    manualSeed = random.randint(1, 10000)
    cudnn.benchmark = True

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=4)
    device = torch.device("cuda:0")

    ngpu = 1
    nz = 100
    ngf = 64
    ndf = 64
    lr = 0.0002
    beta1 = 0.5

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)


    netG = Generator(ngpu, nz, ngf).to(device)
    netG.apply(weights_init)

    netD = Discriminator(ngpu, ndf).to(device)
    netD.apply(weights_init)


    criterion = nn.BCELoss()

    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    schedulerD = ReduceLROnPlateau(optimizerD, mode='min', factor=0.5, patience=2, verbose=True)
    schedulerG = ReduceLROnPlateau(optimizerG, mode='min', factor=0.5, patience=2, verbose=True)
    print(len(dataloader.dataset))
    writer = SummaryWriter('logs/')
    for epoch in range(opt.nepoch):
        d_error = 0.0
        g_error = 0.0
        for i, data in enumerate(dataloader, 0):
            real_label = 1
            fake_label = 0
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label_noise = (torch.rand(1)[0])/2 - 0.25
            real_label += label_noise
            #print(real_label)
            label = torch.full((batch_size,), real_label,
                            dtype=real_cpu.dtype, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            fake_label += torch.rand(1)[0]/8
            #print(fake_label)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            d_error += errD.item()
            g_error += errG.item()
            # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
            #         % (epoch, opt.nepoch, i, len(dataloader),
            #             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                            '%s/real_%s.png' % (opt.outf, opt.name),
                            normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                            '%s/fake_samples_%s_epoch_%03d_.png' % (opt.outf, opt.name, epoch),
                            normalize=True)
        d_error /= len(dataloader)
        g_error /= len(dataloader)
        #schedulerD.step(d_error)
        #schedulerG.step(g_error)
        writer.add_scalar(f"D_Loss", d_error, epoch)
        writer.add_scalar(f"G_Loss", g_error, epoch)
        print(f"Epoch: {epoch}. G_error: {g_error}. D_error: {d_error}")
            # do checkpointing
        torch.save(netG.main.state_dict(), '%s/netG_epoch_%s_%d.pth' % (opt.outf, opt.name, epoch))
        torch.save(netD.main.state_dict(), '%s/netD_epoch_%s_%d.pth' % (opt.outf, opt.name, epoch))




